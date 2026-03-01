import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernelband.agents.optim_agent import OptimAgent
from kernelband.models.openai_model import OpenAIModel
from kernelband.models.claude_model import ClaudeModel
from kernelband.dataloaders.tritonbench import TritonBench
from kernelband.config import load_config, resolve_tritonbench_paths
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run OptimAgent with TritonBench")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/examples/test_config.yaml",
        help="Path to the configuration file"
    )
    cmd_args = parser.parse_args()

    args = load_config(cmd_args.config)
    args = resolve_tritonbench_paths(args)

    model_type = getattr(args, 'model_type', 'openai').lower()  # Default to 'openai' if not specified
    base_url = getattr(args, 'base_url', None)  # Get base_url from config if available

    if model_type == 'openai':
        model = OpenAIModel(api_key=args.api_key, model_id=args.model_id, base_url=base_url)
    elif model_type == 'claude':
        model = ClaudeModel(api_key=args.api_key, model_id=args.model_id, base_url=base_url)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported types: 'openai', 'gemini', 'claude'")

    dataset = TritonBench(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path, 
                          py_interpreter=args.py_interpreter, 
                          golden_metrics=args.golden_metrics,
                          perf_ref_folder=args.perf_ref_folder,
                          perf_G_path=args.perf_G_path,
                          result_path=args.result_path,
                          target_kernels=args.target_kernels)

    mab_config = getattr(args, 'mab_config', None)

    generation_budget_per_kernel = getattr(args, 'generation_budget_per_kernel', None)

    if args.max_iteration > 0 and generation_budget_per_kernel is not None:
        raise ValueError(
            "Configuration error: 'max_iteration' and 'generation_budget_per_kernel' cannot be set simultaneously. "
            "Please use ONLY ONE mode:\n"
            "  - Iteration mode: Set max_iteration > 0, do not set generation_budget_per_kernel\n"
            "  - Budget mode: Set generation_budget_per_kernel > 0, set max_iteration = 0"
        )

    max_tokens = getattr(args, 'max_tokens', 16384)

    keep_perf_logs = getattr(args, 'keep_perf_logs', False)

    agent = OptimAgent(model=model, dataset=dataset, corpus_path=args.corpus_path,
                       mem_file=args.mem_file,
                       mab_config=mab_config, generation_budget_per_kernel=generation_budget_per_kernel,
                       max_tokens=max_tokens,
                       keep_perf_logs=keep_perf_logs)

    thread_num = getattr(args, 'thread_num', 3)  # Default to 3 if not specified
    agent.run(output_path=args.output_path,
              multi_thread=args.multi_thread,
              thread_num=thread_num,
              iteration_num=args.max_iteration,
              temperature=args.temperature,
              datalen=args.datalen,
              start_iter=args.start_iter,
              start_idx=args.start_idx)


if __name__ == "__main__":
    main()