"""
OptimAgent: MAB-based Kernel Optimization Agent

Implements the paper's hierarchical multi-armed bandit framework:
  Phase 0: Clustering + Hardware Masking
  Phase 1: UCB Selection + Code Generation (ONE per kernel per iteration)
  Phase 2: Correctness Testing
  Phase 3: Performance Testing
  Phase 4: Reward + UCB Update + Frontier Update
  Phase 5: Save Results
"""

from tqdm import tqdm
import os
import re
import json
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from kernelband.agents.reflexion_oneshot import Reflexion_Oneshot
from kernelband.utils.code_utils import extract_function_signatures
from kernelband.memory.memory import MemoryClassMeta
from kernelband.prompts import generation, reflection
from kernelband.prompts.strategy_config import STRATEGY_NAMES, STRATEGY_PROMPTS
from kernelband.mab.state import MABState, KernelEntry
from kernelband.mab.ucb import UCBSelector
from kernelband.mab.clustering import BehavioralClusterer
from kernelband.mab.masking import HardwareMasker
from kernelband.mab.features import extract_behavioral_features, extract_hw_signature
from kernelband.mab.strategies import PAPER_STRATEGY_NAMES, NUM_STRATEGIES, HW_DIM_NAMES, HW_DIM_COUNT, STRATEGY_TARGET_MAP
from loguru import logger


class OptimAgent(Reflexion_Oneshot):
    def __init__(self, model, dataset, corpus_path, mem_file=None,
                 mab_config=None, generation_budget_per_kernel=None,
                 max_tokens=16384, keep_perf_logs=False):
        # Set attributes BEFORE calling super().__init__() because memory_init is called in parent's __init__
        self.generation_budget_per_kernel = generation_budget_per_kernel
        self.max_tokens = max_tokens
        self.keep_perf_logs = keep_perf_logs

        super().__init__(model, dataset, corpus_path, mem_file)

        self.generation_prompt = generation.prompt

        # ==================== GPU Hardware Detection ====================
        self.hardware_info = self._get_hardware_info(gpu_id=0)
        logger.info(f"GPU detected for prompt injection: {self.hardware_info['gpu_name']}")
        logger.info(f"Compute Capability: {self.hardware_info['compute_capability']}")

        # ==================== MAB Configuration ====================
        mab_config = mab_config or {}
        self.ucb_c = mab_config.get('ucb_c', 2.0)
        self.cluster_K = mab_config.get('cluster_K', 3)
        self.cluster_tau = mab_config.get('cluster_tau', 10)
        self.within_cluster_temperature = mab_config.get('within_cluster_temperature', 1.0)

        raw_theta_sat = mab_config.get('theta_sat', 75.0)
        if isinstance(raw_theta_sat, dict):
            for key in raw_theta_sat:
                if key not in HW_DIM_NAMES:
                    raise ValueError(
                        f"Invalid theta_sat dimension '{key}'. "
                        f"Valid dimensions: {list(HW_DIM_NAMES.keys())}"
                    )
            self.theta_sat = np.array([
                float(raw_theta_sat.get('sm', 75.0)),
                float(raw_theta_sat.get('dram', 75.0)),
                float(raw_theta_sat.get('l2', 75.0)),
            ], dtype=np.float64)
        else:
            self.theta_sat = np.full(HW_DIM_COUNT, float(raw_theta_sat), dtype=np.float64)

        raw_target_map = mab_config.get('strategy_target_map', None)
        if raw_target_map is not None:
            self.strategy_target_map = {}
            for strategy, dim_name in raw_target_map.items():
                if strategy not in PAPER_STRATEGY_NAMES:
                    raise ValueError(
                        f"Invalid strategy '{strategy}' in strategy_target_map. "
                        f"Valid strategies: {PAPER_STRATEGY_NAMES}"
                    )
                if dim_name not in HW_DIM_NAMES:
                    raise ValueError(
                        f"Invalid target dimension '{dim_name}' for strategy '{strategy}'. "
                        f"Valid dimensions: {list(HW_DIM_NAMES.keys())}"
                    )
                self.strategy_target_map[strategy] = HW_DIM_NAMES[dim_name]
            # Validate all strategies have a mapping
            for strategy in PAPER_STRATEGY_NAMES:
                if strategy not in self.strategy_target_map:
                    raise ValueError(
                        f"strategy_target_map is missing strategy '{strategy}'. "
                        f"All strategies must have a mapping: {PAPER_STRATEGY_NAMES}"
                    )
        else:
            self.strategy_target_map = STRATEGY_TARGET_MAP

        self.ucb_selector = UCBSelector(
            c=self.ucb_c,
            theta_sat=self.theta_sat,
            temperature=self.within_cluster_temperature,
            strategy_target_map=self.strategy_target_map,
        )
        self.clusterer = BehavioralClusterer(K=self.cluster_K)
        self.masker = HardwareMasker(
            theta_sat=self.theta_sat,
            strategy_target_map=self.strategy_target_map,
        )

        self.ncu_profiler = None
        self.ncu_enabled = mab_config.get('ncu_enabled', True)
        if self.ncu_enabled:
            try:
                from kernelband.pruning.ncu_profiler import NCUProfiler
                ncu_cache_dir = mab_config.get('ncu_cache_dir', './ncu_profiler_cache')
                os.makedirs(ncu_cache_dir, exist_ok=True)
                self.ncu_profiler = NCUProfiler(
                    cache_dir=ncu_cache_dir,
                    py_folder=self.dataset.py_folder if hasattr(self.dataset, 'py_folder') else None
                )
                logger.info(f"NCU profiler initialized for hardware masking")
            except Exception as e:
                logger.warning(f"NCU profiler initialization failed: {e}")
                self.ncu_profiler = None
                self.ncu_enabled = False

        logger.info(
            f"MAB config: ucb_c={self.ucb_c}, "
            f"theta_sat=[SM={self.theta_sat[0]:.1f}, DRAM={self.theta_sat[1]:.1f}, L2={self.theta_sat[2]:.1f}], "
            f"K={self.cluster_K}, tau={self.cluster_tau}, "
            f"ncu_enabled={self.ncu_enabled}"
        )


    def memory_init(self, mem_file=None):
        """
        Initialize memory with MAB state for each kernel.

        Args:
            mem_file: previous stored memories, which can be loaded to continue run
        """
        class Memory(metaclass=MemoryClassMeta, field_names=["ps",
                                                             "call_err_msg",
                                                             "exe_err_msg",
                                                             "reflection",
                                                             "function_signatures",
                                                             "oneshot",
                                                             "strategy",    # Main code strategy (backward compat)
                                                             "branches",    # This iteration's results
                                                             "best_code",   # Global best [code, ms, eff, speedup, chain]
                                                             "_generation_budget",  # Budget mode only
                                                             "_mab_state",  # MAB state summary dict
                                                             "_mab_state_obj"]):  # MAB state object (runtime only)
            pass

        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)
            assert len(input_mems) == len(self.dataset), \
                f"expect {len(self.dataset)} samples, but got {len(input_mems)} instead"

            # Check mode compatibility
            first_mem_key = list(input_mems.keys())[0]
            has_budget_field = "_generation_budget" in input_mems[first_mem_key]
            is_budget_mode = self.generation_budget_per_kernel is not None

            if is_budget_mode and not has_budget_field:
                raise ValueError(
                    f"Mode mismatch: Current run uses budget mode, "
                    f"but memory file '{mem_file}' was created in iteration mode."
                )
            elif not is_budget_mode and has_budget_field:
                raise ValueError(
                    f"Mode mismatch: Current run uses iteration mode, "
                    f"but memory file '{mem_file}' was created in budget mode."
                )

        for ps in self.dataset.problem_states:
            if ps.label:
                fs_mem = extract_function_signatures(ps.label)
            else:
                fs_mem = None

            if mem_file is None:
                os_mem = self.instruction_retriever.query(ps.instruction)[0]

                baseline_code = ps.label if ps.label else ""
                baseline_ms, baseline_efficiency, baseline_pass_perf = None, None, False

                if baseline_code and self.dataset.perf_ref_folder:
                    try:
                        path_baseline = os.path.join(
                            self.dataset.perf_ref_folder,
                            ps.filename[:-3] + ".json"
                        )
                        if os.path.exists(path_baseline):
                            _, baseline_efficiency, baseline_ms = self.dataset.calculate(
                                path_baseline,
                                path_ref=path_baseline
                            )
                            baseline_pass_perf = True
                            ms_str = f"{baseline_ms:.2f}" if baseline_ms is not None else "N/A"
                            logger.info(f"{ps.filename}: Baseline with perf data (ms={ms_str})")
                        else:
                            logger.warning(f"{ps.filename}: Baseline perf file not found at {path_baseline}")
                    except Exception as e:
                        logger.warning(f"{ps.filename}: Failed to read baseline performance: {e}")

                if not baseline_code:
                    logger.warning(f"{ps.filename}: No baseline code, will use oneshot as fallback")

                initial_mab_state = None
                if baseline_code:
                    initial_frontier_entry = {
                        "code": baseline_code,
                        "ms": baseline_ms,
                        "efficiency": baseline_efficiency,
                        "speedup": 1.0 if baseline_pass_perf else None,
                        "strategy": "baseline",
                        "strategy_chain": ["baseline"],
                        "pass_call": True,
                        "pass_exe": True,
                        "pass_perf": baseline_pass_perf,
                        "call_err_msg": None,
                        "exe_err_msg": None,
                    }
                    initial_mab_state = {
                        "K": self.cluster_K if hasattr(self, 'cluster_K') else 3,
                        "S": NUM_STRATEGIES,
                        "total_t": 0,
                        "mu_hat": np.full((self.cluster_K if hasattr(self, 'cluster_K') else 3, NUM_STRATEGIES), 0.5).tolist(),
                        "N": np.ones((self.cluster_K if hasattr(self, 'cluster_K') else 3, NUM_STRATEGIES)).tolist(),
                        "masks": np.ones((self.cluster_K if hasattr(self, 'cluster_K') else 3, NUM_STRATEGIES)).tolist(),
                        "last_cluster_iter": -1,
                        "frontier": [initial_frontier_entry],
                        "best_entry": initial_frontier_entry if baseline_pass_perf else None,
                    }

                tmp_mem = Memory(
                    ps=ps,
                    call_err_msg=None,
                    exe_err_msg=None,
                    reflection=None,
                    function_signatures=fs_mem,
                    oneshot=os_mem["code"],
                    strategy=None,
                    branches={},
                    best_code=[],
                    _generation_budget=self.generation_budget_per_kernel if self.generation_budget_per_kernel is not None else None,
                    _mab_state=initial_mab_state,
                    _mab_state_obj=None,
                )
            else:
                input_mem = input_mems[ps.filename]

                required_fields = ["best_code", "branches"]
                for field in required_fields:
                    if field not in input_mem:
                        raise ValueError(
                            f"Memory file format error for {ps.filename}: missing required field '{field}'."
                        )

                tmp_mem = Memory(
                    ps=ps,
                    call_err_msg=input_mem.get("call_err_msg"),
                    exe_err_msg=input_mem.get("exe_err_msg"),
                    reflection=input_mem.get("reflection"),
                    function_signatures=fs_mem,
                    oneshot=input_mem["oneshot"],
                    strategy=input_mem.get("strategy"),
                    branches=input_mem.get("branches", {}),
                    best_code=input_mem.get("best_code", []),
                    _generation_budget=input_mem.get("_generation_budget"),
                    _mab_state=input_mem.get("_mab_state"),
                    _mab_state_obj=None,
                )

            self.memories.append(tmp_mem)

    def write_memories(self, file_path):
        output_dict = {}
        with open(file_path, "w") as f:
            for mem in self.memories:
                output = {
                    "call_err_msg": str(mem.call_err_msg) if mem.call_err_msg else None,
                    "exe_err_msg": str(mem.exe_err_msg) if mem.exe_err_msg else None,
                    "reflection": mem.reflection,
                    "oneshot": mem.oneshot,
                    "strategy": mem.strategy,
                    "branches": mem.branches,
                    "best_code": mem.best_code,
                }
                if self.generation_budget_per_kernel is not None:
                    output["_generation_budget"] = mem._generation_budget
                if hasattr(mem, '_mab_state_obj') and mem._mab_state_obj is not None:
                    output["_mab_state"] = mem._mab_state_obj.to_summary_dict()
                elif mem._mab_state is not None:
                    output["_mab_state"] = mem._mab_state
                output_dict[mem.ps.filename] = output
            json.dump(output_dict, f, indent=2)

    # ==================== MAB Initialization ====================

    def _init_mab_state(self, mem):
        """Initialize MABState for a kernel from saved state or baseline."""
        K = self.cluster_K
        S = NUM_STRATEGIES

        if mem._mab_state is not None and isinstance(mem._mab_state, dict):
            if "frontier" in mem._mab_state:
                # New format: frontier is serialized inside _mab_state
                state = MABState.from_summary_dict(mem._mab_state)
            else:
                # Old format backward compat: _mab_state has no frontier key.
                # Reconstruct frontier from "beam" field in the memory file if present.
                state = MABState.from_summary_dict(mem._mab_state)
                if hasattr(mem, 'ps') and mem.ps.label:
                    # Use baseline code as minimal frontier
                    entry = KernelEntry(
                        code=mem.ps.label,
                        strategy="baseline",
                        strategy_chain=["baseline"],
                        pass_call=True, pass_exe=True, pass_perf=False,
                    )
                    entry.phi = extract_behavioral_features(entry.code, entry.ms)
                    entry.cluster_id = 0
                    state.frontier.append(entry)

            # Recover best_entry from best_code if not already set
            if state.best_entry is None and mem.best_code and len(mem.best_code) >= 4:
                state.best_entry = KernelEntry(
                    code=mem.best_code[0],
                    ms=mem.best_code[1],
                    efficiency=mem.best_code[2],
                    speedup=mem.best_code[3],
                    strategy_chain=mem.best_code[4] if len(mem.best_code) > 4 else [],
                    pass_call=True, pass_exe=True, pass_perf=True,
                )
            return state

        state = MABState(K=K, S=S)

        if hasattr(mem, 'ps') and mem.ps.label:
            baseline_code = mem.ps.label
            # Look for pre-built frontier data from memory_init
            # (stored as initial_mab_state during fresh init, but _mab_state was None means no baseline)
            entry = KernelEntry(
                code=baseline_code,
                strategy="baseline",
                strategy_chain=["baseline"],
                pass_call=True,
                pass_exe=True,
                pass_perf=False,
            )
            entry.phi = extract_behavioral_features(entry.code, entry.ms)
            entry.cluster_id = 0
            state.frontier.append(entry)

        if mem.best_code and len(mem.best_code) >= 4:
            state.best_entry = KernelEntry(
                code=mem.best_code[0],
                ms=mem.best_code[1],
                efficiency=mem.best_code[2],
                speedup=mem.best_code[3],
                strategy_chain=mem.best_code[4] if len(mem.best_code) > 4 else [],
                pass_call=True, pass_exe=True, pass_perf=True,
            )
        elif state.frontier:
            # Use best from frontier
            perf_entries = [e for e in state.frontier if e.pass_perf and e.ms is not None]
            if perf_entries:
                perf_entries.sort(key=lambda e: e.ms)
                state.best_entry = perf_entries[0]

        return state

    # ==================== Code Generation Methods ====================

    def _call_llm_and_extract_code(self, prompt_text, temperature, context_info=""):
        """
        Call LLM and extract code from response.

        Expected format: Markdown code block (```python...```)
        Fallback: JSON format ({"code": "..."}) for backward compatibility
        """
        msg = [{"role": "user", "content": prompt_text}]

        try:
            response = self.model.generate(msg, temperature=temperature, max_tokens=self.max_tokens)
        except Exception as e:
            logger.error(f"Failed to call LLM{' for ' + context_info if context_info else ''}: {e}")
            return ""

        code = self._extract_markdown_code_block(response)
        if code:
            return code

        code = self._extract_json_code(response)
        if code:
            logger.debug(f"Extracted code from JSON fallback{' for ' + context_info if context_info else ''}")
            return code

        logger.warning(f"Failed to extract code{' for ' + context_info if context_info else ''}")
        logger.debug(f"Response preview: {response[:500]}")
        return ""

    def _extract_markdown_code_block(self, text):
        """Extract code from markdown code block."""
        pattern = r'```(?:python|py|triton)?\s*\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        pattern_open = r'```(?:python|py|triton)?\s*\n(.+)'
        match = re.search(pattern_open, text, re.DOTALL)
        if match:
            logger.debug("Extracted from truncated code block (no closing ```)")
            return match.group(1).strip()

        return ""

    def _extract_json_code(self, text):
        """Extract code from JSON format (backward compatibility)."""
        try:
            match = re.search(r'\{.+\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_obj = json.loads(json_str)
                if isinstance(json_obj, dict) and "code" in json_obj:
                    return json_obj["code"].strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return ""

    # ==================== Main MAB Run Loop ====================

    def run(self, output_path=None, multi_thread=True, thread_num=3, datalen=None,
            iteration_num=0, temperature=1.0, start_idx=0, gpu_id=0, start_iter=0):
        """
        MAB-based optimization run loop.

        Per iteration, per kernel: ONE (cluster, strategy) selection via Masked UCB,
        ONE code generation, correctness testing, performance testing, reward update.
        """
        if iteration_num > 0 and self.generation_budget_per_kernel is not None:
            raise ValueError(
                "Configuration error: 'iteration_num' and 'generation_budget_per_kernel' cannot be set simultaneously."
            )

        data_len = datalen if datalen else len(self.dataset)

        SAFEGUARD_MAX_ITER = 1000
        is_budget_mode = self.generation_budget_per_kernel is not None

        for mem in self.memories[start_idx:(start_idx + data_len)]:
            mem._mab_state_obj = self._init_mab_state(mem)

        for iter_idx in range(start_iter, start_iter + SAFEGUARD_MAX_ITER):
            if is_budget_mode:
                budgets_remaining = [mem._generation_budget for mem in self.memories[start_idx:(start_idx + data_len)]]
                if all(b <= 0 for b in budgets_remaining):
                    logger.info("All kernels exhausted generation budget, stopping")
                    break
            else:
                if iter_idx >= start_iter + iteration_num:
                    break

            logger.info(f"\n{'='*60}")
            logger.info(f"=== Iteration {iter_idx} (MAB Mode) ===")
            logger.info(f"{'='*60}")

            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{iter_idx}{extension}"
                mem_output_path = f"{root}_mem_{iter_idx}.json"
                tmp_dir = f"{root}_tmp"
                exe_dir_base = f"{root}_pass_exe"
                perf_result_dir = f"{root}_perf_results"
                perf_log_dir = f"{root}_perf_logs_iter{iter_idx}"
            else:
                iter_path = None
                mem_output_path = None
                tmp_dir = "tmp"
                exe_dir_base = "pass_exe"
                perf_result_dir = "perf_results"
                perf_log_dir = f"perf_logs_iter{iter_idx}"

            # ==================== PHASE 0: CLUSTERING + MASKING ====================
            logger.info(f"\n[PHASE 0] Clustering + Hardware Masking")

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                mab = mem._mab_state_obj

                for entry in mab.frontier:
                    if entry.phi is None:
                        entry.phi = extract_behavioral_features(entry.code, entry.ms)

                # Re-cluster if conditions met
                # NOTE (D1): Paper (Alg. 1, line 7) triggers at fixed intervals (t mod tau = 0).
                # This implementation triggers relative to last_cluster_iter, so the first
                # clustering may happen before iteration tau if frontier grows fast. Functionally
                # similar but not identical to the paper.
                if self.clusterer.should_recluster(
                    len(mab.frontier), iter_idx, mab.last_cluster_iter, self.cluster_tau
                ):
                    assignments, centers = self.clusterer.cluster(mab.frontier)
                    mab.cluster_centers = centers
                    mab.last_cluster_iter = iter_idx
                    logger.info(f"{mem.ps.filename}: Re-clustered {len(mab.frontier)} kernels")
                elif not any(e.cluster_id is not None for e in mab.frontier):
                    # First time: assign all to cluster 0
                    for entry in mab.frontier:
                        entry.cluster_id = 0

                cluster_hw_sigs = {}
                if self.ncu_enabled and self.ncu_profiler:
                    for cluster_id in range(mab.K):
                        representative = self.clusterer.get_cluster_representative(
                            mab.frontier, cluster_id, centers=mab.cluster_centers
                        )
                        if representative and representative.code:
                            if representative.hw_sig is not None:
                                cluster_hw_sigs[cluster_id] = representative.hw_sig
                            else:
                                try:
                                    profiling_result = self.ncu_profiler.profile_best_code(
                                        kernel_name=mem.ps.filename,
                                        best_code=representative.code,
                                        iteration=iter_idx
                                    )
                                    hw_sig = extract_hw_signature(profiling_result)
                                    if hw_sig is not None:
                                        representative.hw_sig = hw_sig
                                        cluster_hw_sigs[cluster_id] = hw_sig
                                except Exception as e:
                                    logger.warning(
                                        f"{mem.ps.filename}: NCU profiling failed for cluster {cluster_id}: {e}"
                                    )

                mab.masks = self.masker.compute_masks(mab, cluster_hw_sigs)

                # Then mask empty clusters (must come AFTER compute_masks to avoid being overwritten)
                for cluster_id in range(mab.K):
                    has_members = any(e.cluster_id == cluster_id for e in mab.frontier)
                    if not has_members:
                        mab.masks[cluster_id, :] = 0.0

                active_count = int(mab.masks.sum())
                total_arms = mab.K * mab.S
                logger.debug(
                    f"{mem.ps.filename}: Masks: {active_count}/{total_arms} arms active"
                )

            # ==================== PHASE 1: UCB SELECTION + CODE GENERATION ====================
            logger.info(f"\n[PHASE 1] UCB Selection + Code Generation")

            generation_tasks = []

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                mab = mem._mab_state_obj
                mem._new_codes = []

                if is_budget_mode and mem._generation_budget <= 0:
                    logger.info(f"{mem.ps.filename}: Budget exhausted, skipping")
                    continue

                if len(mab.frontier) == 0:
                    logger.info(f"{mem.ps.filename}: Empty frontier, generating from oneshot")
                    # Generate one code per strategy from oneshot
                    strategies_to_gen = STRATEGY_NAMES[:]
                    if is_budget_mode:
                        strategies_to_gen = strategies_to_gen[:max(0, mem._generation_budget)]

                    for strategy in strategies_to_gen:
                        generation_tasks.append({
                            "mem": mem,
                            "strategy": strategy,
                            "parent_entry": None,
                            "cluster_id": 0,
                            "strategy_idx": PAPER_STRATEGY_NAMES.index(strategy),
                            "mode": "optimization",
                            "is_first_iteration": True,
                        })
                    continue

                # Check if any frontier entry has debug needs (pass_exe=False)
                debug_entries = [e for e in mab.frontier if not e.pass_exe]
                if debug_entries and len(debug_entries) == len(mab.frontier):
                    # All entries failing - use debug mode for the first one
                    entry = debug_entries[0]
                    generation_tasks.append({
                        "mem": mem,
                        "strategy": entry.strategy,
                        "parent_entry": entry,
                        "cluster_id": entry.cluster_id or 0,
                        "strategy_idx": PAPER_STRATEGY_NAMES.index(entry.strategy) if entry.strategy in PAPER_STRATEGY_NAMES else 0,
                        "mode": "debug",
                    })
                    continue

                cluster_id, strategy_idx, strategy_name = self.ucb_selector.select(mab)

                parent_entry = self.ucb_selector.sample_from_cluster(
                    mab.frontier, cluster_id, strategy_name
                )

                if parent_entry is None:
                    # Fallback: use best entry or first frontier entry
                    parent_entry = mab.best_entry if mab.best_entry else mab.frontier[0]

                logger.info(
                    f"{mem.ps.filename}: UCB -> cluster={cluster_id}, "
                    f"strategy={strategy_name}, parent_ms={parent_entry.ms}"
                )

                generation_tasks.append({
                    "mem": mem,
                    "strategy": strategy_name,
                    "parent_entry": parent_entry,
                    "cluster_id": cluster_id,
                    "strategy_idx": strategy_idx,
                    "mode": "optimization",
                    "is_first_iteration": (iter_idx == 0),
                })

            logger.info(f"Total generation tasks: {len(generation_tasks)}")

            def process_generation_task(task):
                """Process a single code generation task."""
                try:
                    if task["mode"] == "optimization":
                        parent = task["parent_entry"]
                        if parent is None:
                            current_code = {
                                "code": task["mem"].oneshot,
                                "ms": 0,
                                "strategy": "oneshot"
                            }
                        else:
                            current_code = {
                                "code": parent.code,
                                "ms": parent.ms,
                                "strategy": parent.strategy,
                                "efficiency": parent.efficiency,
                            }

                        code = self.generate_optimization_code(
                            task["mem"],
                            task["strategy"],
                            current_code,
                            temperature,
                            task.get("is_first_iteration", False)
                        )

                        task["mem"]._new_codes.append({
                            "code": code,
                            "strategy": task["strategy"],
                            "cluster_id": task["cluster_id"],
                            "strategy_idx": task["strategy_idx"],
                            "parent_entry": task["parent_entry"],
                            "mode": "optimization",
                        })
                    else:
                        parent = task["parent_entry"]
                        reflection_text = self.generate_reflection_for_debug(
                            task["mem"],
                            parent.code,
                            parent.pass_call,
                            parent.call_err_msg,
                            parent.exe_err_msg,
                            temperature
                        )
                        fixed_code = self.generate_debug_code(
                            task["mem"],
                            parent.code,
                            parent.call_err_msg,
                            parent.exe_err_msg,
                            reflection_text,
                            temperature
                        )
                        task["mem"]._new_codes.append({
                            "code": fixed_code,
                            "strategy": task["strategy"],
                            "cluster_id": task["cluster_id"],
                            "strategy_idx": task["strategy_idx"],
                            "parent_entry": task["parent_entry"],
                            "mode": "debug",
                        })
                except Exception as e:
                    logger.error(f"Generation failed for {task['mem'].ps.filename}/{task['strategy']}: {e}")

            if multi_thread and len(generation_tasks) > 0:
                with ThreadPoolExecutor(max_workers=thread_num) as executor:
                    list(tqdm(
                        executor.map(process_generation_task, generation_tasks),
                        total=len(generation_tasks),
                        desc="Generating codes"
                    ))
            else:
                for task in tqdm(generation_tasks, desc="Generating codes"):
                    process_generation_task(task)

            if is_budget_mode:
                for mem in self.memories[start_idx:(start_idx + data_len)]:
                    count = len(mem._new_codes)
                    if count > 0:
                        old_budget = mem._generation_budget
                        mem._generation_budget -= count
                        logger.info(
                            f"{mem.ps.filename}: Generated {count} codes, "
                            f"budget {old_budget} -> {mem._generation_budget}"
                        )

            total_new_codes = sum(len(mem._new_codes) for mem in self.memories[start_idx:(start_idx + data_len)])
            if total_new_codes == 0:
                logger.info(f"No codes generated in iteration {iter_idx}, skipping testing phases")
            else:
                # ==================== PHASE 2: CORRECTNESS TESTING ====================
                logger.info(f"\n[PHASE 2] Correctness Testing")

                for mem in tqdm(self.memories[start_idx:(start_idx + data_len)], desc="Testing correctness"):
                    kernel_name = mem.ps.filename[:-3]

                    for code_idx, new_code_info in enumerate(mem._new_codes):
                        code = new_code_info["code"]
                        code_exe_dir = f"{exe_dir_base}_{kernel_name}_code{code_idx}"

                        if os.path.exists(code_exe_dir):
                            shutil.rmtree(code_exe_dir)
                        os.makedirs(code_exe_dir, exist_ok=True)

                        try:
                            pass_call, pass_exe, _, call_stderr, _, exe_stderr, kernel_metadata = \
                                self.dataset.test_opt_correctness(
                                    code, mem.ps.filename, tmp_dir, exe_dir=code_exe_dir
                                )

                            new_code_info["pass_call"] = pass_call
                            new_code_info["pass_exe"] = pass_exe
                            new_code_info["kernel_metadata"] = kernel_metadata

                            if not pass_call:
                                new_code_info["call_err_msg"] = call_stderr
                                new_code_info["exe_err_msg"] = exe_stderr
                            elif not pass_exe:
                                new_code_info["call_err_msg"] = None
                                new_code_info["exe_err_msg"] = exe_stderr
                            else:
                                new_code_info["call_err_msg"] = None
                                new_code_info["exe_err_msg"] = None

                        except Exception as e:
                            logger.error(f"Failed to test {mem.ps.filename} code{code_idx}: {e}")
                            new_code_info["pass_call"] = False
                            new_code_info["pass_exe"] = False
                            new_code_info["call_err_msg"] = f"Test failed: {e}"
                            new_code_info["exe_err_msg"] = f"Test failed: {e}"

                # ==================== PHASE 3: PERFORMANCE TESTING (Serial) ====================
                logger.info(f"\n[PHASE 3] Performance Testing (Serial)")

                codes_to_test = []
                for mem in self.memories[start_idx:(start_idx + data_len)]:
                    kernel_name = mem.ps.filename[:-3]
                    for code_idx, new_code_info in enumerate(mem._new_codes):
                        if new_code_info.get("pass_exe", False):
                            codes_to_test.append((mem, kernel_name, code_idx))

                logger.info(f"Codes to test: {len(codes_to_test)}")

                for mem, kernel_name, code_idx in tqdm(codes_to_test, desc="Running perf tests"):
                    code_exe_dir = f"{exe_dir_base}_{kernel_name}_code{code_idx}"
                    code_perf_dir = f"{perf_result_dir}_{kernel_name}_code{code_idx}"
                    code_script_dir = os.path.join(tmp_dir, f"perf_gen_{kernel_name}_code{code_idx}")
                    code_log_dir = f"{perf_log_dir}_{kernel_name}_code{code_idx}"

                    if not os.path.exists(code_exe_dir) or not os.listdir(code_exe_dir):
                        logger.warning(f"{kernel_name} code{code_idx} directory empty, skipping")
                        continue

                    try:
                        if os.path.exists(code_script_dir):
                            shutil.rmtree(code_script_dir)

                        self.dataset.write_perf_file(
                            input_folder_path=code_exe_dir,
                            results_path=code_perf_dir,
                            tmp_dir=code_script_dir
                        )
                        self.dataset.run_perf_scripts(
                            gpu_id=gpu_id,
                            script_dir=code_script_dir,
                            log_dir=code_log_dir
                        )
                    except Exception as e:
                        logger.error(f"Performance test failed for {kernel_name} code{code_idx}: {e}")

                # Parse performance results
                logger.info(f"\n[PHASE 3] Parsing performance results")
                for mem in tqdm(self.memories[start_idx:(start_idx + data_len)], desc="Parsing perf results"):
                    kernel_name = mem.ps.filename[:-3]

                    for code_idx, new_code_info in enumerate(mem._new_codes):
                        if not new_code_info.get("pass_exe", False):
                            continue

                        code_perf_dir = f"{perf_result_dir}_{kernel_name}_code{code_idx}"
                        path_gen = os.path.join(code_perf_dir, mem.ps.filename[:-3] + ".json")

                        if os.path.exists(path_gen):
                            try:
                                path_ref = None
                                if self.dataset.perf_ref_folder:
                                    path_ref = os.path.join(self.dataset.perf_ref_folder, mem.ps.filename[:-3] + ".json")
                                    if not os.path.exists(path_ref):
                                        path_ref = None
                                    elif os.path.exists(path_ref):
                                        with open(path_ref, 'r') as f:
                                            baseline_data = json.load(f)
                                            if len(baseline_data) == 0:
                                                path_ref = None

                                speedup, efficiency, ms = self.dataset.calculate(path_gen, path_ref=path_ref)
                                new_code_info["ms"] = ms
                                new_code_info["efficiency"] = efficiency
                                new_code_info["speedup"] = speedup
                                new_code_info["pass_perf"] = True
                            except Exception as e:
                                logger.error(f"Failed to parse perf for {kernel_name} code{code_idx}: {e}")
                                new_code_info["pass_perf"] = False
                        else:
                            logger.warning(f"Performance result not found for {kernel_name} code{code_idx}")
                            new_code_info["pass_perf"] = False

            # ==================== PHASE 4: REWARD + UCB UPDATE + FRONTIER UPDATE ====================
            logger.info(f"\n[PHASE 4] Reward + UCB Update + Frontier Update")

            for mem in tqdm(self.memories[start_idx:(start_idx + data_len)], desc="Updating MAB state"):
                mab = mem._mab_state_obj

                for new_code_info in mem._new_codes:
                    code = new_code_info.get("code", "")
                    strategy = new_code_info.get("strategy", "")
                    cluster_id = new_code_info.get("cluster_id", 0)
                    strategy_idx = new_code_info.get("strategy_idx", 0)
                    parent_entry = new_code_info.get("parent_entry")

                    new_entry = KernelEntry(
                        code=code,
                        ms=new_code_info.get("ms"),
                        efficiency=new_code_info.get("efficiency"),
                        speedup=new_code_info.get("speedup"),
                        strategy=strategy,
                        strategy_chain=(parent_entry.strategy_chain + [strategy]) if parent_entry else [strategy],
                        pass_call=new_code_info.get("pass_call", False),
                        pass_exe=new_code_info.get("pass_exe", False),
                        pass_perf=new_code_info.get("pass_perf", False),
                        call_err_msg=new_code_info.get("call_err_msg"),
                        exe_err_msg=new_code_info.get("exe_err_msg"),
                    )
                    new_entry.phi = extract_behavioral_features(
                        code, new_entry.ms,
                        kernel_metadata=new_code_info.get("kernel_metadata")
                    )
                    new_entry.cluster_id = cluster_id

                    # Add to frontier if valid (pass_exe at minimum)
                    if new_entry.pass_exe and code:
                        mab.frontier.append(new_entry)

                    # UCB update (paper Algorithm 1, lines 14-18)
                    # NOTE (D2): UCB statistics are updated after EVERY generation attempt.
                    # Successful attempts use the latency-based reward; failed attempts
                    # (compilation error, correctness failure) receive reward=0. This ensures
                    # that repeatedly failing strategies have their mu_hat driven down and
                    # their exploration bonus reduced, so the algorithm rotates to untried
                    # strategies instead of getting stuck.
                    if new_entry.pass_perf and new_entry.ms is not None:
                        parent_ms = parent_entry.ms if parent_entry else None
                        reward = UCBSelector.compute_reward(parent_ms, new_entry.ms)

                        if mab.best_entry is None or (
                            mab.best_entry.ms is not None and new_entry.ms < mab.best_entry.ms
                        ):
                            mab.best_entry = new_entry
                            logger.info(
                                f"{mem.ps.filename}: New best code "
                                f"(ms={new_entry.ms:.4f}, strategy={strategy})"
                            )
                    else:
                        reward = 0.0

                    UCBSelector.update(mab, cluster_id, strategy_idx, reward)
                    logger.debug(
                        f"{mem.ps.filename}: reward={reward:.4f}, "
                        f"mu_hat[{cluster_id},{strategy_idx}]="
                        f"{mab.mu_hat[cluster_id, strategy_idx]:.4f}, "
                        f"N={int(mab.N[cluster_id, strategy_idx])}"
                    )

                if mab.best_entry:
                    mem.best_code = [
                        mab.best_entry.code,
                        mab.best_entry.ms,
                        mab.best_entry.efficiency,
                        mab.best_entry.speedup,
                        mab.best_entry.strategy_chain,
                    ]

                logger.debug(
                    f"{mem.ps.filename}: Frontier={len(mab.frontier)}, "
                    f"best_ms={mab.best_entry.ms if mab.best_entry else 'N/A'}"
                )

            # ==================== PHASE 5: SAVE RESULTS ====================
            logger.info(f"\n[PHASE 5] Save Results")

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                mem.branches = {}
                for new_code_info in mem._new_codes:
                    key = f"mab_{new_code_info['strategy']}"
                    mem.branches[key] = [
                        new_code_info["code"],
                        new_code_info.get("ms"),
                        new_code_info.get("efficiency"),
                        None  # No reflection for optimization mode
                    ]

                mab = mem._mab_state_obj
                if len(mem.best_code) > 0:
                    mem.ps.solution = mem.best_code[0]
                elif mab and mab.best_entry:
                    mem.ps.solution = mab.best_entry.code

                if mab and mab.best_entry:
                    mem.strategy = mab.best_entry.strategy

                del mem._new_codes

            if output_path is not None:
                self.dataset.write_file(iter_path, start_idx=start_idx, datalen=data_len)
                self.write_memories(mem_output_path)

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                kernel_name = mem.ps.filename[:-3]
                num_codes = len(mem.branches)
                for code_idx in range(num_codes):
                    for d in [
                        f"{exe_dir_base}_{kernel_name}_code{code_idx}",
                        f"{perf_result_dir}_{kernel_name}_code{code_idx}",
                    ]:
                        if os.path.exists(d):
                            shutil.rmtree(d)
                    if not self.keep_perf_logs:
                        log_d = f"{perf_log_dir}_{kernel_name}_code{code_idx}"
                        if os.path.exists(log_d):
                            shutil.rmtree(log_d)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            logger.info(f"\n{'='*60}")
            logger.info(f"=== Iteration {iter_idx} Complete ===")
            logger.info(f"{'='*60}\n")

    # ==================== Code Generation (kept from original) ====================

    def generate_optimization_code(self, mem, strategy, parent_code, temperature, is_first_iteration):
        """Generate code for a specific strategy branch (Optimization Mode)."""
        tab = "\n"
        fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        text = self.generation_prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text,
            **self.hardware_info
        )

        text += f"\n\n[OPTIMIZATION STRATEGY: {strategy}]\n"
        text += STRATEGY_PROMPTS[strategy]

        if is_first_iteration:
            text += f"\n\nHere is a similar code example for coding style reference:\n{mem.oneshot}\n"

            if parent_code:
                baseline_ms = parent_code.get("ms")
                baseline_eff = parent_code.get("efficiency")

                if baseline_ms is not None:
                    text += f"""
Here is the baseline code (correct implementation) to optimize:
{parent_code["code"]}

Baseline Performance: {baseline_ms:.2f} ms
Baseline Efficiency: {baseline_eff if baseline_eff is not None else "N/A"}

Your task: Apply **{strategy}** optimization techniques to improve upon this baseline.
Analyze both the style reference and the baseline to generate better code.
"""
                else:
                    text += f"""
Here is the baseline code (correct implementation) to optimize:
{parent_code["code"]}

Note: Baseline performance data is not available.

Your task: Apply **{strategy}** optimization techniques to improve this code.
Focus on correctness first, then optimize for performance.
"""
            else:
                text += f"""
Note: No baseline code available. The example above is for coding style reference only.
Generate a correct and optimized implementation using **{strategy}** techniques.
"""
        else:
            current_ms = parent_code.get("ms")
            current_ms_str = f"{current_ms:.2f} ms" if current_ms is not None else "N/A (performance test failed)"
            text += f"""

Here is the current code to optimize:
{parent_code["code"]}

Current Performance: {current_ms_str}
Current Strategy: {parent_code["strategy"]}

Your task: Apply **{strategy}** optimization techniques to improve this code.
"""

        text += f"""

**Output Format:**
Output your complete optimized code in a Python code block:
```python
<your complete code here>
```

Focus on **{strategy}** optimization techniques.
Do not change function names, parameter names, counts, or order.
"""

        return self._call_llm_and_extract_code(text, temperature, f"{strategy} / {mem.ps.filename}")

    def generate_debug_code(self, mem, current_code, call_err_msg, exe_err_msg, reflection_text, temperature=0):
        """Generate solution in debug mode (fix errors)."""
        tab = "\n"
        fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        text = self.generation_prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text,
            **self.hardware_info
        )

        one_shot = self.code_retriever.query(current_code)[0]["code"]
        text += f"\nHere is an example snippet of code: {one_shot}"
        text += f"\nPrevious attempt implementation:{current_code}"

        if call_err_msg:
            text += f"\nTest messages for previous attempt:{call_err_msg}"
        if exe_err_msg:
            text += f"\nTest messages for correctness check of previous attempt:{exe_err_msg}"
        if reflection_text:
            text += f"\nReflection on previous attempt:{reflection_text}"

        text += """

**Output Format:**
Output your corrected code in a Python code block:
```python
<your complete corrected code here>
```

Generate code that can run directly without errors.
Do not change function names, parameter names, counts, or order.
"""

        code = self._call_llm_and_extract_code(text, temperature, mem.ps.filename)
        return code if code else ""

    def generate_reflection_for_debug(self, mem, code, pass_call, call_err_msg, exe_err_msg, temperature):
        """Generate reflection for debug mode."""
        if pass_call:
            reflect_txt = reflection.prompt_exe.format(
                problem=mem.ps.instruction,
                solution=code,
                call_test_result="succeed",
                exe_test_result=exe_err_msg,
                **self.hardware_info
            )
        else:
            reflect_txt = reflection.prompt.format(
                problem=mem.ps.instruction,
                solution=code,
                test_result=call_err_msg,
                **self.hardware_info
            )

        reflect_msg = [{"role": "user", "content": reflect_txt}]
        result = self.model.generate(reflect_msg, temperature=temperature)
        return result

    # ==================== Hardware Info ====================

    def _get_hardware_info(self, gpu_id=0):
        """Extract GPU hardware specifications for prompt injection."""
        import torch
        from kernelband.dataloaders.gpu_specs import get_arch_name, get_hardware_limits

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using conservative hardware defaults")
            return {
                "gpu_name": "CPU (No GPU detected)",
                "compute_capability": "N/A",
                "architecture": "N/A",
                "max_shared_mem_per_block_kb": "48",
                "max_shared_mem_per_block_bytes": "49152",
                "sm_count": "N/A",
                "max_threads_per_block": "1024",
                "warp_size": "32",
                "total_memory_gb": "N/A",
            }

        try:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            props = torch.cuda.get_device_properties(gpu_id)

            major, minor = torch.cuda.get_device_capability(gpu_id)
            arch_name = get_arch_name((major, minor))
            compute_capability = f"{major}.{minor} - {arch_name}"
            architecture = arch_name
            warp_size = props.warp_size

            limits = get_hardware_limits((major, minor))
            max_shared_mem = limits['max_shared_mem_per_block']
            max_threads = limits['max_threads_per_block']

            logger.info(f"Detected NVIDIA GPU: {gpu_name} (CC {major}.{minor})")

            hardware_info = {
                "gpu_name": gpu_name,
                "compute_capability": compute_capability,
                "architecture": architecture,
                "max_shared_mem_per_block_kb": str(max_shared_mem // 1024),
                "max_shared_mem_per_block_bytes": str(max_shared_mem),
                "sm_count": str(props.multi_processor_count),
                "max_threads_per_block": str(max_threads),
                "warp_size": str(warp_size),
                "total_memory_gb": f"{props.total_memory / (1024**3):.1f}",
            }

            logger.info(f"GPU Hardware Specs:")
            logger.info(f"  - Max Shared Memory per Block: {hardware_info['max_shared_mem_per_block_kb']} KB")
            logger.info(f"  - SM Count: {hardware_info['sm_count']}")
            logger.info(f"  - Max Threads per Block: {hardware_info['max_threads_per_block']}")
            logger.info(f"  - Warp Size: {hardware_info['warp_size']}")

            return hardware_info

        except Exception as e:
            logger.error(f"Failed to get GPU hardware info: {e}")
            return {
                "gpu_name": "Unknown GPU",
                "compute_capability": "N/A",
                "architecture": "N/A",
                "max_shared_mem_per_block_kb": "48",
                "max_shared_mem_per_block_bytes": "49152",
                "sm_count": "N/A",
                "max_threads_per_block": "1024",
                "warp_size": "32",
                "total_memory_gb": "N/A",
            }
