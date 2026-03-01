#!/usr/bin/env python3
"""
Helper script to list and filter TritonBench kernels for batch evaluation.

Usage:
    python list_kernels.py --all                    # List all kernels
    python list_kernels.py --difficulty 1 2         # List kernels with difficulty 1-2
    python list_kernels.py --range 0 10             # List kernels at index 0-9
    python list_kernels.py --yaml --difficulty 1 2  # Output YAML format for config
    python list_kernels.py --config my.yaml --difficulty 1 2  # Update config file directly

    # Subset support (from sample_tritonbench_subset.py output)
    python list_kernels.py --subset tritonbench_subset_50.json           # List subset kernels
    python list_kernels.py --subset tritonbench_subset_50.json --stats   # Show subset statistics
    python list_kernels.py --subset tritonbench_subset_50.json --config my.yaml  # Apply subset to config
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

def load_kernels(json_path):
    """Load kernel information from TritonBench JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_subset(subset_path, full_kernels=None):
    """
    Load kernel subset from a sampled subset JSON file.

    Args:
        subset_path: Path to the subset JSON file (e.g., tritonbench_subset_50.json)
        full_kernels: Optional list of full kernel data for enrichment

    Returns:
        Tuple of (kernels_list, metadata_dict)
    """
    with open(subset_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})

    # Check if detailed info is available
    if 'detailed' in data:
        kernels = data['detailed']
    elif 'kernels' in data:
        # Only filenames available, try to enrich from full data
        kernel_files = set(data['kernels'])
        if full_kernels:
            kernels = [k for k in full_kernels if k.get('file') in kernel_files]
        else:
            # Create minimal kernel entries
            kernels = [{'file': f} for f in data['kernels']]
    else:
        raise ValueError(f"Invalid subset file format: {subset_path}")

    return kernels, metadata

def filter_by_difficulty(kernels, difficulties):
    """Filter kernels by difficulty levels."""
    # Normalize difficulties to strings for comparison
    diff_set = set(str(d) for d in difficulties)
    return [k for k in kernels if str(k.get('difficulty', '')) in diff_set]

def filter_by_range(kernels, start_idx, end_idx):
    """Filter kernels by index range."""
    return kernels[start_idx:end_idx]

def print_kernels(kernels, output_yaml=False):
    """Print kernels in list or YAML format."""
    if output_yaml:
        print("target_kernels:")
        for kernel in kernels:
            print(f'  - "{kernel.get("file", "unknown")}"')
    else:
        print(f"\nTotal: {len(kernels)} kernels\n")
        for i, kernel in enumerate(kernels):
            file = kernel.get('file', 'unknown')
            diff = kernel.get('difficulty', '?')
            print(f"{i:3d}. {file:<50s} (difficulty: {diff})")

def print_statistics(kernels, title="TritonBench Kernel Statistics"):
    """Print kernel statistics by difficulty."""
    difficulty_count = {}
    for kernel in kernels:
        diff = kernel.get('difficulty', 'unknown')
        difficulty_count[diff] = difficulty_count.get(diff, 0) + 1

    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    print(f"\nTotal kernels: {len(kernels)}")
    print("\nKernels by difficulty:")
    for diff in sorted(difficulty_count.keys()):
        print(f"  Difficulty {diff}: {difficulty_count[diff]:3d} kernels")
    print("="*70 + "\n")


def print_subset_statistics(kernels, metadata=None):
    """Print detailed statistics for a sampled subset."""
    print("\n" + "="*70)
    print("  Sampled Subset Statistics")
    print("="*70)

    # Metadata info
    if metadata:
        print(f"\nSource: {metadata.get('source', 'Unknown')}")
        print(f"Original size: {metadata.get('original_size', '?')}")
        print(f"Subset size: {metadata.get('subset_size', len(kernels))}")
        print(f"Sampling method: {metadata.get('sampling_method', 'Unknown')}")
        print(f"Random seed: {metadata.get('seed', '?')}")

    print(f"\nTotal kernels in subset: {len(kernels)}")

    # Difficulty distribution
    difficulty_count = defaultdict(int)
    for kernel in kernels:
        diff = kernel.get('difficulty', 'unknown')
        difficulty_count[diff] += 1

    print("\nDifficulty distribution:")
    for diff in sorted(difficulty_count.keys()):
        pct = difficulty_count[diff] / len(kernels) * 100
        print(f"  Level {diff}: {difficulty_count[diff]:3d} kernels ({pct:5.1f}%)")

    # Category distribution (if available)
    category_count = defaultdict(int)
    for kernel in kernels:
        cat = kernel.get('category')
        if cat:
            category_count[cat] += 1

    if category_count:
        print(f"\nCategory distribution ({len(category_count)} categories):")
        for cat in sorted(category_count.keys()):
            pct = category_count[cat] / len(kernels) * 100
            print(f"  {cat:<25s}: {category_count[cat]:3d} ({pct:5.1f}%)")

    print("="*70 + "\n")

def update_config_file(config_path, kernels):
    """Update target_kernels field in YAML config file."""
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        return False

    # Read the config file
    with open(config_path, 'r') as f:
        content = f.read()

    # Generate the new target_kernels section
    kernel_list = [f'  - "{k.get("file", "unknown")}"' for k in kernels]
    new_target_kernels = "target_kernels:\n" + "\n".join(kernel_list)

    # Find and replace the target_kernels section
    # Pattern matches:
    # target_kernels: null
    # OR
    # target_kernels:
    #   - "kernel1.py"
    #   - "kernel2.py"
    #   ...

    # First try to match target_kernels: null
    pattern_null = r'target_kernels:\s*null'
    if re.search(pattern_null, content):
        new_content = re.sub(pattern_null, new_target_kernels, content)
    else:
        # Match target_kernels with list items
        # This pattern matches the target_kernels line and all following lines that start with whitespace + dash
        pattern_list = r'target_kernels:(?:\n\s+-\s*"[^"]*")*'
        if re.search(pattern_list, content):
            new_content = re.sub(pattern_list, new_target_kernels, content)
        else:
            print(f"⚠️  Warning: Could not find 'target_kernels' field in config file")
            print(f"   Please add the following to your config manually:")
            print(f"\n{new_target_kernels}\n")
            return False

    # Write back to file
    with open(config_path, 'w') as f:
        f.write(new_content)

    print(f"✅ Updated {config_path}")
    print(f"   Set target_kernels to {len(kernels)} kernels")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='List and filter TritonBench kernels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show statistics
  python list_kernels.py --stats

  # List easy kernels (difficulty 1-2)
  python list_kernels.py --difficulty 1 2

  # List first 10 kernels
  python list_kernels.py --range 0 10

  # Generate YAML config for easy kernels
  python list_kernels.py --yaml --difficulty 1 2 > easy_kernels.yaml

  # Generate config for kernels 10-20
  python list_kernels.py --yaml --range 10 20 > kernels_10_20.yaml

  # Use sampled subset (from sample_tritonbench_subset.py)
  python list_kernels.py --subset scripts/tritonbench_subset_50.json --stats
  python list_kernels.py --subset scripts/tritonbench_subset_50.json --yaml
  python list_kernels.py --subset scripts/tritonbench_subset_50.json --config my.yaml
        """
    )

    # Default path relative to scripts directory — try both formats
    default_json = Path(__file__).parent.parent / 'data/TritonBench/statistics.json'
    if not default_json.is_file():
        alt = Path(__file__).parent.parent / 'data/TritonBench/data/TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json'
        if alt.is_file():
            default_json = alt

    parser.add_argument('--json', type=str,
                       default=str(default_json),
                       help='Path to TritonBench JSON file')
    parser.add_argument('--subset', type=str,
                       help='Use a sampled subset JSON file (e.g., tritonbench_subset_50.json)')
    parser.add_argument('--all', action='store_true',
                       help='List all kernels')
    parser.add_argument('--difficulty', type=str, nargs='+',
                       help='Filter by difficulty level (e.g., --difficulty 1 2)')
    parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'),
                       help='Filter by index range (e.g., --range 0 10)')
    parser.add_argument('--stats', action='store_true',
                       help='Show kernel statistics')
    parser.add_argument('--yaml', action='store_true',
                       help='Output in YAML format for config file')
    parser.add_argument('--config', type=str,
                       help='Update target_kernels in the specified config file')

    args = parser.parse_args()

    # Handle subset mode
    if args.subset:
        subset_path = Path(args.subset)
        if not subset_path.is_absolute():
            # Try relative to current dir, then relative to scripts dir
            if not subset_path.exists():
                subset_path = Path(__file__).parent / args.subset
            if not subset_path.exists():
                subset_path = Path(__file__).parent.parent / args.subset

        if not subset_path.exists():
            print(f"Error: Could not find subset file at {args.subset}")
            return 1

        # Load full kernels for enrichment (optional)
        try:
            full_kernels = load_kernels(args.json)
        except FileNotFoundError:
            full_kernels = None

        try:
            subset_kernels, metadata = load_subset(str(subset_path), full_kernels)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Invalid subset file format: {e}")
            return 1

        # Show subset statistics
        if args.stats:
            print_subset_statistics(subset_kernels, metadata)
            return 0

        # Apply additional filters to subset
        filtered_kernels = subset_kernels

        if args.difficulty:
            filtered_kernels = filter_by_difficulty(filtered_kernels, args.difficulty)
            if not args.yaml and not args.config:
                print(f"\n🔍 Filtering subset by difficulty: {', '.join(args.difficulty)}")

        if args.range:
            start, end = args.range
            filtered_kernels = filter_by_range(filtered_kernels, start, end)
            if not args.yaml and not args.config:
                print(f"\n🔍 Filtering subset by range: [{start}, {end})")

        # Update config file if requested
        if args.config:
            success = update_config_file(args.config, filtered_kernels)
            if not success:
                return 1

            # Print summary
            print(f"\n📋 Kernels from subset applied to config ({len(filtered_kernels)} total):")
            for i, kernel in enumerate(filtered_kernels[:10]):
                print(f"   {i+1}. {kernel.get('file', 'unknown')}")
            if len(filtered_kernels) > 10:
                print(f"   ... and {len(filtered_kernels)-10} more")
            return 0

        # Default: show stats if no other action, otherwise list/yaml
        if not args.all and not args.difficulty and not args.range and not args.yaml:
            print_subset_statistics(subset_kernels, metadata)
            return 0

        # Print results
        print_kernels(filtered_kernels, args.yaml)
        return 0

    # Original mode: load from full TritonBench JSON
    try:
        kernels = load_kernels(args.json)
    except FileNotFoundError:
        print(f"Error: Could not find JSON file at {args.json}")
        print("Please run from project root or specify correct path with --json")
        return 1

    # Show statistics
    if args.stats or (not args.all and not args.difficulty and not args.range):
        print_statistics(kernels)
        return 0

    # Filter kernels
    filtered_kernels = kernels

    if args.difficulty:
        filtered_kernels = filter_by_difficulty(filtered_kernels, args.difficulty)
        if not args.yaml:
            print(f"\n🔍 Filtering by difficulty: {', '.join(args.difficulty)}")

    if args.range:
        start, end = args.range
        filtered_kernels = filter_by_range(filtered_kernels, start, end)
        if not args.yaml and not args.config:
            print(f"\n🔍 Filtering by range: [{start}, {end})")

    # Update config file if requested
    if args.config:
        if not args.difficulty and not args.range:
            print("⚠️  Warning: No filters specified. This will set target_kernels to ALL kernels.")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0

        success = update_config_file(args.config, filtered_kernels)
        if not success:
            return 1

        # Also print what was set
        print(f"\n📋 Kernels set in config ({len(filtered_kernels)} total):")
        for i, kernel in enumerate(filtered_kernels[:10]):
            print(f"   {i+1}. {kernel.get('file', 'unknown')}")
        if len(filtered_kernels) > 10:
            print(f"   ... and {len(filtered_kernels)-10} more")
        return 0

    # Print results
    print_kernels(filtered_kernels, args.yaml)
    return 0

if __name__ == '__main__':
    exit(main())
