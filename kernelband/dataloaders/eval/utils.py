import ast
import os
import subprocess
from random import randint
from tqdm import tqdm
from shutil import copyfile
import datetime
import json
import numpy as np

# Path to correctness.py script (resolved relative to this file)
_CORRECTNESS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness.py")

## Implementation from https://arxiv.org/pdf/2107.03374
def passk(n, c, k):
    if n -c < k: return 1.0
    return 1 - np.prod(
        1 - k/ np.arange(
            n-c+1, n+1
        )
    )

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_temp_file(prefix='temp_code'):
    temp_file_name = f'{prefix}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name = f'{prefix}_{randint(999, 999999)}.py'
    return temp_file_name


def extract_code_from_llm_output(response):
    if "```" not in response:
        return response
    import re
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return response

def get_fname_difficulty_from_label(label):
    tritonbench_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "TritonBench")
    candidates = [
        os.path.join(tritonbench_root, "statistics.json"),
        os.path.join(tritonbench_root, "data",
                     "TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json"),
    ]
    triton_root = None
    for path in candidates:
        if os.path.isfile(path):
            triton_root = path
            break
    if triton_root is None:
        return None, None
    with open(triton_root, 'r') as f:
        data = json.load(f)
        for item in data:
            if item['output'] == label:
                return item['file'], item['difficulty']
    return None, None

def process_code(code: str):
    if "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
    
    try:
        tree = ast.parse(code)
        imports = []
        function_definitions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                function_code = ast.unparse(node)
                function_definitions.append(function_code)

        return "\n".join(imports) + "\n\n" + "\n".join(function_definitions)

    except:
        return code


_METADATA_EXTRACTION_SNIPPET = '''
# === Kernel Metadata Extraction ===
import json as _json
try:
    import triton as _triton
    _metadata = {}
    for _name, _obj in list(globals().items()):
        if isinstance(_obj, _triton.JITFunction) and hasattr(_obj, 'cache') and _obj.cache:
            for _key, _compiled in _obj.cache.items():
                _metadata = {
                    'n_regs': getattr(_compiled, 'n_regs', None),
                    'shared': getattr(_compiled, 'shared', None),
                    'num_warps': getattr(_compiled, 'num_warps', None),
                }
                break
            break
    if _metadata:
        with open(__file__ + '.metadata.json', 'w') as _f:
            _json.dump(_metadata, _f)
except Exception:
    pass
'''


def _read_metadata(gen_file):
    """Try to read kernel metadata extracted during test execution."""
    metadata_path = gen_file + '.metadata.json'
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            os.remove(metadata_path)
            return metadata
    except Exception:
        pass
    # Clean up even if read fails
    try:
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    except Exception:
        pass
    return None


def code_call_exec_success_allclose(code, fname, py_folder, temp_root="tmp2", atol=1e-4, rtol=1e-4, timeout=2*60, verbose=False, compile_only=False):
    """
    Test code compilation and optionally correctness.

    Args:
        code: The generated code to test
        fname: Filename for the test
        py_folder: Path to the folder containing reference Python files
        temp_root: Root directory for temporary files
        atol: Absolute tolerance for allclose comparison
        rtol: Relative tolerance for allclose comparison
        timeout: Timeout in seconds for subprocess calls
        verbose: Enable verbose output
        compile_only: If True, only perform compilation check and return early (skip correctness test)

    Returns:
        (call_status, exec_status, call_stdout, call_stderr, exe_stdout, exe_stderr, kernel_metadata)

        kernel_metadata: dict with actual compilation metadata (n_regs, shared, num_warps)
                        or None if extraction failed.
        When compile_only=True:
            exec_status, exe_stdout, exe_stderr are always None
    """
    tmp_gen_folder = os.path.join(temp_root, "gen")
    os.makedirs(tmp_gen_folder, exist_ok=True)

    triton_root = py_folder
    triton_file = os.path.join(triton_root, fname)

    gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
    gen_file = os.path.join(tmp_gen_folder, gen_file)

    hash_line = "#"*146

    with open(triton_file, 'r') as f:
        lines = f.readlines()
        iL = None
        for iL, line in enumerate(lines):
            if line.strip() == hash_line:
                break
        else:
            # Hash line not found
            return None, None, None, f"Error: Could not find separator line (146 #'s) in file {fname}", None, None, None
        test_code_lines = lines[iL+1:]
        test_code_lines_procs = test_code_lines

    code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)
    code += _METADATA_EXTRACTION_SNIPPET

    with open(gen_file, 'w') as f:
        f.write(code)

    try:
        result_call = subprocess.run([f'python3 {gen_file}'], capture_output=True, text=True, timeout=timeout, shell=True)
        call_status = result_call.returncode == 0

        # Read metadata after call test (extracted during execution)
        kernel_metadata = _read_metadata(gen_file)

        # Early return for compile-only mode (skip correctness test)
        if compile_only:
            return call_status, None, result_call.stdout, result_call.stderr, None, None, kernel_metadata

        # Check for correctness
        result_corr = subprocess.run([f'python3 {_CORRECTNESS_SCRIPT} --gen_file {gen_file} --ref_file {triton_file} --atol {atol} --rtol {rtol}'], capture_output=True, text=True, timeout=timeout, shell=True)
        stdout_corr = result_corr.stdout
        stderr_corr = result_corr.stderr

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"File: {fname} timed out!")
        return None, None, None, "Time out", None, None, None
    except Exception as e:
        if verbose:
            print(f"File: {fname}, Execution error: {e}")
        return None, None, None, str(e), None, None, None
    finally:
        pass

    with open(gen_file+".stdout", 'w') as f:
        f.write(stdout_corr)

    with open(gen_file+".stderr", 'w') as f:
        f.write(stderr_corr)

    if result_corr.returncode != 0:
        if verbose:
            print(f"Error in generated code: {stderr_corr}")
        return call_status, None, result_call.stdout, result_call.stderr, stdout_corr, stderr_corr, kernel_metadata
    else:
        if verbose:
            print(f"Success in generated code: {stdout_corr}")
        _, exec_status, gen_stdout, gen_stderr = stdout_corr.split("*#*#")
        return call_status, exec_status, result_call.stdout, result_call.stderr, gen_stdout, gen_stderr, kernel_metadata

    

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def green_or_red(status):
    if status:
        return bcolors.OKGREEN
    else:
        return bcolors.FAIL

def color_end():
    return bcolors.ENDC

def bool_colorize(status):
    if status:
        return bcolors.OKGREEN + str(status) + bcolors.ENDC
    else:
        return bcolors.FAIL + str(status) + bcolors.ENDC