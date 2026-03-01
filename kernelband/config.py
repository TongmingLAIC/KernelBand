import os
import logging
import yaml
from argparse import Namespace

logger = logging.getLogger(__name__)

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)


# Mapping from new-format config fields to old-format (original TritonBench repo) paths.
# Each entry: (config_attr, new_suffix, old_suffix)
_PATH_MAP = [
    ("statis_path",       "data/TritonBench/statistics.json",
                          "data/TritonBench/data/TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json"),
    ("py_folder",         "data/TritonBench/kernels",
                          "data/TritonBench/data/TritonBench_G_v1"),
    ("instruction_path",  "data/TritonBench/statistics.json",
                          "data/TritonBench/data/TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json"),
    ("corpus_path",       "data/TritonBench/corpus/train_crawl.json",
                          "data/TritonBench/data/train_crawl.json"),
    ("golden_metrics",    "data/TritonBench/perf_metrics/golden_metrics",
                          "data/TritonBench/performance_metrics/perf_G/golden_metrics"),
    ("perf_G_path",       "data/TritonBench/perf_metrics",
                          "data/TritonBench/performance_metrics/perf_G"),
]


def resolve_tritonbench_paths(args):
    """Auto-detect TritonBench directory format and remap config paths if needed.

    The original TritonBench repository uses a different directory layout than the
    refactored one.  This function checks whether the configured ``py_folder``
    exists on disk.  If it does, no remapping is performed.  Otherwise it looks
    for the old-format indicator directory and remaps every path that is still set
    to its new-format default to the corresponding old-format path.
    """
    py_folder = getattr(args, "py_folder", None)
    if py_folder and os.path.exists(py_folder):
        return args  # new format present – nothing to do

    # Check for old-format indicator
    old_kernels_dir = os.path.join("data", "TritonBench", "data", "TritonBench_G_v1")
    if not os.path.isdir(old_kernels_dir):
        return args  # neither format found – leave paths as-is

    logger.info("Detected original TritonBench repository layout; remapping paths")

    for attr, new_suffix, old_suffix in _PATH_MAP:
        current = getattr(args, attr, None)
        if current is None:
            continue
        # Only remap if the current value matches the new-format default
        # (normalise to avoid trailing-slash mismatches)
        if os.path.normpath(current) == os.path.normpath(new_suffix):
            setattr(args, attr, old_suffix)
            logger.info("  %s: %s -> %s", attr, new_suffix, old_suffix)

    return args