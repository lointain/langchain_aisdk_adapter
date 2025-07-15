#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol Tests Runner

This script helps run different categories of protocol tests:
- basic: Text (0), Error (3), Finish (d)
- tools: Tool Call (9), Tool Result (a), Tool Start (b)
- steps: Step Finish (e), Step Start (f) - for agent workflows

"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def print_usage():
    print("Protocol Tests Runner")
    print("Usage: python run_tests.py [category]")
    print("")
    print("Categories:")
    print("  basic     - Text (0), Data (2), Error (3), Finish (d) protocols")
    print("  tools     - Tool Call (9), Tool Result (a), Tool Start (b) protocols")
    print("  steps     - Step Finish (e), Step Start (f) protocols (for agents)")
    print("  all       - Run all tests")
    print("")
    print("Examples:")
    print("  python run_tests.py basic")
    print("  python run_tests.py tools")
    print("  python run_tests.py all")


async def run_test_file(test_file):
    """Run a single test file"""
    print(f"\n{'='*60}")
    print(f"Running: {test_file.name}")
    print(f"{'='*60}")
    
    try:
        # Import and run the test
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # If the module has a main function, run it
        if hasattr(test_module, '__main__'):
            # The test file should handle its own execution
            pass
        
        return True
    except Exception as e:
        print(f"❌ Error running {test_file.name}: {e}")
        return False


# Test file mapping
TEST_FILES = {
    "basic": [
        "test_protocol_0d_text_finish.py",
        "test_protocol_2_langgraph_monitoring.py",
        "test_protocol_3_error.py"  # Keep separate as it tests error conditions
    ],
    "tools": [
        "test_protocol_9ab_tools.py"
    ],
    "steps": [
        "test_protocol_ef_steps.py"
    ]
}


def run_category_tests(category):
    """Run tests for a specific category"""
    test_dir = Path(__file__).parent / category
    
    if not test_dir.exists():
        print(f"❌ Category '{category}' not found")
        return False
    
    # Use predefined test files instead of globbing
    if category not in TEST_FILES:
        print(f"❌ Category '{category}' not supported")
        return False
    
    test_files = []
    for filename in TEST_FILES[category]:
        test_file = test_dir / filename
        if test_file.exists():
            test_files.append(test_file)
        else:
            print(f"⚠️  Test file not found: {filename}")
    
    if not test_files:
        print(f"❌ No test files found in '{category}' category")
        return False
    
    print(f"Found {len(test_files)} test(s) in '{category}' category:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    success_count = 0
    for test_file in test_files:
        try:
            # Run the test file as a subprocess to avoid import conflicts
            import subprocess
            result = subprocess.run([sys.executable, str(test_file)], 
                                  capture_output=True, text=True, cwd=str(test_file.parent))
            
            print(f"\n{'='*60}")
            print(f"Running: {test_file.name}")
            print(f"{'='*60}")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            if result.returncode == 0:
                print(f"✅ {test_file.name} PASSED")
                success_count += 1
            else:
                print(f"❌ {test_file.name} FAILED (exit code: {result.returncode})")
                
        except Exception as e:
            print(f"❌ Error running {test_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Category '{category}' Results: {success_count}/{len(test_files)} tests passed")
    print(f"{'='*60}")
    
    return success_count == len(test_files)


def main():
    if len(sys.argv) != 2:
        print_usage()
        return
    
    category = sys.argv[1].lower()
    
    if category == "all":
        categories = ["basic", "tools", "steps"]
        total_success = True
        
        for cat in categories:
            success = run_category_tests(cat)
            total_success = total_success and success
        
        print(f"\n{'='*60}")
        if total_success:
            print("🎉 ALL TESTS PASSED")
        else:
            print("💥 SOME TESTS FAILED")
        print(f"{'='*60}")
        
    elif category in ["basic", "tools", "steps"]:
        success = run_category_tests(category)
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Unknown category: {category}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    import importlib.util
    main()