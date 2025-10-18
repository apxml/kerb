#!/usr/bin/env python3
"""Run all example files in docs/examples/ to ensure they execute without errors.

This script:
- Discovers all Python files in docs/examples/
- Runs each example file in a subprocess
- Captures and reports any errors
- Provides a summary of results

Usage:
    python scripts/run_all_examples.py
    python scripts/run_all_examples.py --verbose
    python scripts/run_all_examples.py --stop-on-error
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


def find_example_files(examples_dir: Path, subpackage: str = None) -> List[Path]:
    """Find all Python example files in the examples directory.
    
    Args:
        examples_dir: Path to the examples directory
        subpackage: Optional subdirectory name to filter examples (e.g., "agent")
        
    Returns:
        List of Path objects for Python files, sorted by path
    """
    example_files = []
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        return example_files
    
    # Determine search directory
    search_dir = examples_dir / subpackage if subpackage else examples_dir
    
    if subpackage and not search_dir.exists():
        print(f"❌ Subpackage directory not found: {search_dir}")
        return example_files
    
    # Recursively find all .py files
    for py_file in search_dir.rglob("*.py"):
        # Skip __init__.py files
        if py_file.name == "__init__.py":
            continue
        example_files.append(py_file)
    
    return sorted(example_files)


def run_example(file_path: Path, verbose: bool = False) -> Tuple[bool, str, str]:
    """Run a single example file and capture output.
    
    Args:
        file_path: Path to the example file
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout per example
            cwd=file_path.parent  # Run in the file's directory
        )
        
        success = result.returncode == 0
        
        if verbose:
            if result.stdout:
                print(f"  Output:\n{result.stdout}")
            if result.stderr:
                print(f"  Errors:\n{result.stderr}")
        
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Timeout: Example took longer than 30 seconds"
    except Exception as e:
        return False, "", f"Exception running example: {str(e)}"


def get_relative_path(file_path: Path, base_path: Path) -> str:
    """Get a readable relative path for display.
    
    Args:
        file_path: Full path to the file
        base_path: Base path to make relative to
        
    Returns:
        Relative path as string
    """
    try:
        return str(file_path.relative_to(base_path))
    except ValueError:
        return str(file_path)


def print_summary(results: Dict[str, Tuple[bool, str, str]], 
                  examples_dir: Path,
                  verbose: bool = False):
    """Print a summary of all test results.
    
    Args:
        results: Dictionary mapping file paths to (success, stdout, stderr)
        examples_dir: Base examples directory
        verbose: Whether to show verbose output
    """
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Group by category (subdirectory)
    categories = defaultdict(list)
    for file_path, (success, stdout, stderr) in results.items():
        # Get category from path (first subdirectory under examples)
        rel_path = Path(file_path).relative_to(examples_dir)
        category = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        categories[category].append((file_path, success, stdout, stderr))
    
    total_passed = sum(1 for _, (success, _, _) in results.items() if success)
    total_failed = len(results) - total_passed
    
    # Print results by category
    for category in sorted(categories.keys()):
        files = categories[category]
        passed = sum(1 for _, success, _, _ in files if success)
        failed = len(files) - passed
        
        print(f"\n📁 {category}: {passed} passed, {failed} failed")
        
        for file_path, success, stdout, stderr in sorted(files):
            rel_path = get_relative_path(Path(file_path), examples_dir)
            status = "✅" if success else "❌"
            print(f"  {status} {rel_path}")
            
            if not success and not verbose:
                # Show error message for failed tests (if not verbose, as verbose already showed it)
                if stderr:
                    # Only show first few lines of error
                    error_lines = stderr.strip().split('\n')
                    if len(error_lines) > 5:
                        print(f"      Error: {error_lines[-5]}")
                        print(f"             {error_lines[-1]}")
                    else:
                        for line in error_lines:
                            print(f"      {line}")
    
    # Overall summary
    print("\n" + "-"*80)
    print(f"Total: {len(results)} examples")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 All examples passed!")
    else:
        print(f"\n⚠️  {total_failed} example(s) failed")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run all example files in docs/examples/ to ensure they execute without errors"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output including stdout/stderr for each example"
    )
    parser.add_argument(
        "--stop-on-error", "-s",
        action="store_true",
        help="Stop execution on first error"
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Path to examples directory (default: docs/examples/)"
    )
    parser.add_argument(
        "--subpackage",
        type=str,
        default=None,
        help="Run examples from specific subpackage only (e.g., 'agent' for examples/agent/*)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    examples_dir = args.examples_dir or project_root / "docs" / "examples"
    
    if args.subpackage:
        print(f"🔍 Searching for examples in: {examples_dir}/{args.subpackage}")
    else:
        print(f"🔍 Searching for examples in: {examples_dir}")
    
    # Find all example files
    example_files = find_example_files(examples_dir, args.subpackage)
    
    if not example_files:
        print("❌ No example files found!")
        sys.exit(1)
    
    print(f"📝 Found {len(example_files)} example files")
    print()
    
    # Run each example
    results = {}
    for i, file_path in enumerate(example_files, 1):
        rel_path = get_relative_path(file_path, examples_dir)
        print(f"[{i}/{len(example_files)}] Running {rel_path}...", end=" ")
        sys.stdout.flush()
        
        success, stdout, stderr = run_example(file_path, args.verbose)
        results[str(file_path)] = (success, stdout, stderr)
        
        if success:
            print("✅")
        else:
            print("❌")
            if args.stop_on_error:
                print(f"\n⛔ Stopping on error (--stop-on-error flag set)")
                if stderr:
                    print(f"Error output:\n{stderr}")
                sys.exit(1)
    
    # Print summary
    print_summary(results, examples_dir, args.verbose)
    
    # Exit with error code if any examples failed
    failed_count = sum(1 for _, (success, _, _) in results.items() if not success)
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
