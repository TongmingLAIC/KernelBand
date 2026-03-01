import json
import os
import ast
import subprocess
from random import randint
from tqdm import tqdm
import signal
from multiprocessing import Pool, Lock, Value
from kernelband.dataloaders.problem_state import ProblemState
from kernelband.dataloaders.eval.utils import code_call_exec_success_allclose
from kernelband.dataloaders.gpu_specs import get_gpu_specs
import torch
from loguru import logger



class TritonBench:
    def __init__(self,
                 statis_path,
                 py_folder,
                 instruction_path,
                 golden_metrics,
                 py_interpreter,
                 perf_G_path,
                 perf_ref_folder=None,
                 result_path=None,
                 target_kernels=None
                 ):
        self.statis_path = statis_path
        self.py_folder = py_folder
        self.instruction_path = instruction_path
        self.golden_metrics_folder = golden_metrics
        self.py_interpreter = py_interpreter
        self.perf_G_path = perf_G_path
        self.result_path = result_path

        if perf_ref_folder is None and perf_G_path:
            gpu_name = self._get_gpu_name()
            if gpu_name:
                self.perf_ref_folder = os.path.join(perf_G_path, f"{gpu_name}_golden_metrics")
            else:
                self.perf_ref_folder = None
        else:
            self.perf_ref_folder = perf_ref_folder

        gpu_name = self._get_gpu_name()
        if gpu_name:
            gpu_specs = get_gpu_specs(gpu_name)
            if gpu_specs:
                self.gpu_peak_bandwidth, self.gpu_peak_tflops = gpu_specs
                logger.info(f"Detected GPU: {gpu_name}")
                logger.info(f"GPU Specs: {self.gpu_peak_bandwidth} GB/s, {self.gpu_peak_tflops} TFLOPS (FP32)")
            else:
                self.gpu_peak_bandwidth = None
                self.gpu_peak_tflops = None
                logger.warning(f"GPU '{gpu_name}' not found in specs database. Efficiency calculation will be skipped.")
                logger.warning("To add support for this GPU, update dataloaders/gpu_specs.py")
        else:
            self.gpu_peak_bandwidth = None
            self.gpu_peak_tflops = None
            logger.warning("Unable to detect GPU. Efficiency calculation will be skipped.")

        self.problem_states = self.load_ps(result_path, target_kernels)

    def _get_gpu_name(self):
        """Get GPU name for baseline path, returns None if no GPU available"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0).replace(" ", "_")
        except:
            pass
        return None
    
    def load_ps(self, path, target_kernels=None):
        problem_states = []
        if path is None:
            with open(self.instruction_path, "r", encoding='utf-8') as file:
                instructions = json.load(file)
            statis_data = json.loads(open(self.statis_path, 'r', encoding='utf-8').read())

            for line in instructions:
                instruction = line["instruction"]
                label = line["output"]

                g = label.replace("<|im_end|>", "").replace("<|EOT|>", "")
                tmp = False
                for item in statis_data:
                    if g in item["output"]:
                        file = item["file"]
                        tmp = item
                        break
                if target_kernels is not None:
                    if file not in target_kernels:
                        continue
                if tmp:
                    statis_data.remove(tmp)
                elif g[50:220] == 'as tl\n\nif triton.__version__ >= "2.1.0":\n    @triton.jit\n    def _fwd_kernel(\n        Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录':
                        file = "context_attn_nopad.py"
                path = os.path.join(self.py_folder, file)
                assert os.path.exists(path), f"{file} not exist!"
                test_code = open(path, "r", encoding="utf-8").read().split("#"*146)[-1]
                assert "def test_" in  test_code, ""

                problemstate = ProblemState(instruction=instruction,
                                            label=label, 
                                            test_code=test_code, 
                                            filename=file, 
                                            )
                
                problem_states.append(
                    problemstate
                )
        else:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    content = json.loads(line)
                    problem_state = ProblemState(instruction=content["instruction"], 
                                                 label=content["label"], 
                                                 filename=content["filename"],
                                                )
                    if "test_code" in content:
                        problem_state.test_code = content["test_code"]
                    if "predict" in content:
                        problem_state.solution = content["predict"] 
                    problem_states.append(problem_state)
        return problem_states

    def __len__(self):
        return len(self.problem_states)
    
    def write_file(self, file_path, start_idx=0, datalen=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_len = datalen if datalen is not None else len(self)
        with open(file_path, 'w') as f:
            for ps in self.problem_states[start_idx:(start_idx + data_len)]:
                output = {
                    "instruction": ps.instruction,
                    "label": ps.label,
                    "filename": ps.filename,
                }
                if ps.test_code:
                    output["test_code"] = ps.test_code
                if ps.solution:
                    output["predict"] = ps.solution
                else:
                    output["predict"] = ""
                f.write(json.dumps(output) + "\n")
    
    @classmethod
    def run_single_call(cls, ps, tmp_dir="temp", gpu_id=0):
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, ps.filename)
        script_content = ps.solution
        try:
            with open(temp_path, "w") as temp_file:
                temp_file.write(script_content + "\n" + "#" * 146 + "\n" + ps.test_code)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # Run the temporary Python file
            result = subprocess.run(
                ["python", temp_path], 
                capture_output=True, 
                text=True,
                env=env
            )

            success = result.returncode == 0  # Determine if execution was successful

            if success:
                ps.pass_call = True
                return True, None
            else:
                return False, result.stderr

        except Exception as e:
            return False, str(e)
    
    
    def write_perf_file(self, input_folder_path, results_path, tmp_dir):
        """
        input_folder_path: the folder path where codes that pass call and exe tests are stored
        results_path: the folder where perf results (json files) are stored
        tmp_dir: the folder used to store scripts, which are concatenated with codes used to test performance
        """
        
        os.makedirs(tmp_dir, exist_ok=True)
        if os.path.exists(results_path):
            os.system(f'rm -rf {results_path}')
        os.mkdir(results_path)

        tab = ' ' * 4
        performance_utils_path = os.path.join(self.perf_G_path, "performance_utils.py")
        with open(performance_utils_path, 'r') as f:
            performance_utils = f.readlines()
        performance_utils_lines = []
        for line in performance_utils:
            if 'folder_path = ' in line:
                line = tab * 2 + f'folder_path = "{results_path}"\n'
            performance_utils_lines.append(line)
        performance_utils = "".join(performance_utils_lines)
        with open(performance_utils_path, 'w') as f:
            f.write(performance_utils)
        input_file_list = os.listdir(input_folder_path)
        golden_metrics_list = os.listdir(self.golden_metrics_folder)
        for file in input_file_list:
            if file[-3:] == ".py":
                op = file[:-3]
                perf_file_name = op + "_perf.py"
                assert perf_file_name in golden_metrics_list, f"{perf_file_name} not in golden_metrics_list"
                with open(os.path.join(self.golden_metrics_folder, perf_file_name), "r") as f:
                    # golden_metrics = f.read()
                    lines = f.readlines()
                    updated_lines = []
                    for line in lines:
                        if line == "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n":
                            updated_lines.append(f"sys.path.append('{input_folder_path}')\n")
                            updated_lines.append(f"sys.path.append('{self.perf_G_path}')\n")
                        if 'folder_path = ' in line:
                            indent = line[:len(line) - len(line.lstrip())]
                            line = indent + f'folder_path = "{results_path}"\n'
                        line = line.replace("from TritonBench_v1.", "from ")
                        line = line.replace("op_perf.get_do_bench_config()", "op_perf.get_do_bench_config(warmup=100, rep=1000)")
                        updated_lines.append(line)
                    golden_metrics = "".join(updated_lines)
                
                golden_metrics_lines = golden_metrics.split("\n")
                flag = False
                for i in range(len(golden_metrics_lines)):
                    if "input_tensor = self.to_cuda(input_tensor_)" in golden_metrics_lines[i]:
                        index_1 = i
                    if "results.append(result)" in golden_metrics_lines[i]:
                        index_2 = i + 1
                        flag = True
                if flag:
                    
                    for i in range(index_1, index_2):
                        golden_metrics_lines[i] = tab + golden_metrics_lines[i]                
                    golden_metrics_lines.insert(index_1, tab*3 + "try:")
                    golden_metrics_lines.insert(index_2 + 1, tab*3 + "except Exception as e:")
                    golden_metrics_lines.insert(index_2 + 2, tab*4 + 'print(f"Failed to run benchmark for input tensor. Error: {e}")')
                    golden_metrics = "\n".join(golden_metrics_lines)
                
                with open(os.path.join(tmp_dir, perf_file_name), "w") as f:
                    f.write(golden_metrics)
                
    def _run_perf_script(self, args):
        timeout_sec = 600  # 10 mins
        progress_lock = Lock()
        progress = Value('i', 0)

        gpu_id, script, total_scripts, log_dir = args

        script_name = os.path.basename(script)
        log_file = os.path.join(log_dir, f"{script_name}.log")
        err_file = os.path.join(log_dir, f"{script_name}.err")

        # Set CUDA environment variable for GPU selection
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script}"

        with open(log_file, "w") as log, open(err_file, "w") as err:
            process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=err)
        
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # kill the process group
            err.write(f"\n⏱️ Script timed out after {timeout_sec} seconds\n")

        with progress_lock:
            progress.value += 1
            tqdm.write(f"✅ finished {progress.value}/{total_scripts}: {script_name}")
    
   

    def run_perf_scripts_multithread(self, gpu_count, script_dir = "./tmp", log_dir = "./logs"):
        os.makedirs(log_dir, exist_ok=True)

        scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
        scripts = [os.path.join(script_dir, script) for script in scripts]
        total_scripts = len(scripts)  
        
        with Pool(processes=gpu_count) as pool, tqdm(total=total_scripts, desc="Process", ncols=80) as pbar:
            args_list = [(i % gpu_count, scripts[i], total_scripts, log_dir) for i in range(total_scripts)]

            for _ in pool.imap(self._run_perf_script, args_list):
                pbar.update(1)

            pool.close()
            pool.join()
    
    def run_perf_scripts(self, script_dir="./tmp", log_dir="./logs", gpu_id=0):
        """
        Runs performance test scripts on a specified GPU.

        Args:
            script_dir: Directory containing performance test scripts
            log_dir: Directory to store stdout/stderr logs
            gpu_id: GPU device ID
        """
        import time
        os.makedirs(log_dir, exist_ok=True)

        scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
        scripts = [os.path.join(script_dir, script) for script in scripts]
        total_scripts = len(scripts)
        timeout_sec = 600  # 10 mins

        with tqdm(total=total_scripts) as pbar:
            for idx, script in enumerate(scripts):
                script_name = os.path.basename(script)
                log_file = os.path.join(log_dir, f"{script_name}.log")
                err_file = os.path.join(log_dir, f"{script_name}.err")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                start_time = time.time()
                return_code = None
                timed_out = False

                try:
                    with open(log_file, "w") as log_f, open(err_file, "w") as err_f:
                        result = subprocess.run(
                            [self.py_interpreter, script],
                            stdout=log_f,
                            stderr=err_f,
                            env=env,
                            timeout=timeout_sec
                        )
                        return_code = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    with open(err_file, "a") as err_f:
                        err_f.write(f"\n⏱️ Script timed out after {timeout_sec} seconds\n")
                except Exception as e:
                    with open(err_file, "a") as err_f:
                        err_f.write(f"\n❌ Subprocess failed: {e}\n")

                elapsed = time.time() - start_time
                status = "⏱️TIMEOUT" if timed_out else ("✅" if return_code == 0 else f"❌RC={return_code}")
                tqdm.write(f"{status} {idx+1}/{total_scripts}: {script_name} ({elapsed:.1f}s)")
                pbar.update(1)



    def calculate(self, path_gen, path_ref=None):
        get_ms = lambda data: [item["ms"] for item in data]
        get_gbs = lambda data: [item["GB/s"] for item in data]
        get_tflops = lambda data: [item["TFLOPS"] for item in data]
        avg = lambda mss: round(sum(mss[0]) / sum(mss[1]), 4)

        data_gen = json.loads(open(path_gen, 'r', encoding='utf-8').read())
        if path_ref is not None:
            data_ref = json.loads(open(path_ref, 'r', encoding='utf-8').read())
            # If baseline file is empty ([]), treat it as if no baseline was provided
            if len(data_ref) == 0:
                path_ref = None
            else:
                assert len(data_gen) == len(data_ref), ""
                ms_ref = get_ms(data_ref)

        ms_gen = get_ms(data_gen)

        spdup = avg((ms_ref, ms_gen)) if path_ref is not None else None

        # Calculate efficiency using GPU-specific peak performance specs
        if self.gpu_peak_bandwidth is not None and self.gpu_peak_tflops is not None:
            bandwidth_efficiency = round(max(get_gbs(data_gen)) * 100 / self.gpu_peak_bandwidth, 4)
            compute_efficiency = round(max(get_tflops(data_gen)) * 100 / self.gpu_peak_tflops, 4)
            efficiency = max(bandwidth_efficiency, compute_efficiency)
        else:
            # GPU specs not available, skip efficiency calculation
            efficiency = None

        return spdup, efficiency, round(sum(ms_gen)/len(ms_gen), 4)
    
    def test_opt_correctness(self, code, filename, tmp_dir, save_scripts=True, exe_dir="pass_exe", compile_only=False):
        """
        Runs a given Python script on a specified GPU.

        Args:
            code: The generated code to test
            filename: Name of the kernel file
            tmp_dir: Temporary directory for test files
            save_scripts: Whether to save passing scripts
            exe_dir: Directory to save passing scripts
            compile_only: If True, only verify compilation (skip correctness test).

        Returns:
            (pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr, kernel_metadata)

            kernel_metadata: dict with actual compilation metadata (n_regs, shared, num_warps)
                            or None if extraction failed.
            When compile_only=True:
                pass_exe is always False
                exe_stdout, exe_stderr are None
        """
        if compile_only:
            logger.debug(f"{filename}: Running compile-only test (skip correctness)")

        os.makedirs(exe_dir, exist_ok=True)
        call_status, exec_status, call_stdout, call_stderr, exe_stdout, exe_stderr, kernel_metadata = \
            code_call_exec_success_allclose(
                code=code, fname=filename, temp_root=tmp_dir, py_folder=self.py_folder,
                compile_only=compile_only
            )
        pass_call = False
        pass_exe = False
        if "True" in str(call_status):
            pass_call=True
        if not compile_only and "True" in str(exec_status):
            pass_exe=True
            if save_scripts:
                file_exec = os.path.join(exe_dir, filename)
                with open(file_exec, 'w') as f:
                    f.write(code)

        return pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr, kernel_metadata
    
    
    