#!/usr/bin/env python3
"""
Run Validation Framework

Run all project validation tests from the scripts directory.

Usage:
    python scripts/run_validation.py        # Run all validations
    python scripts/run_validation.py logic  # Run logic validator only
    python scripts/run_validation.py test   # Run comprehensive tests only
    python scripts/run_validation.py report # Generate diagnostic report only
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def run_validation(validation_type=None):
    """
    Run validation tests.
    
    Args:
        validation_type (str, optional): Type of validation to run
            - 'logic': Logic validator only
            - 'test': Comprehensive tests only
            - 'report': Diagnostic report only
            - None: Run all
    """
    
    validation_dir = project_root / "validation"
    
    validations = {
        'logic': {
            'script': validation_dir / 'logic_validator.py',
            'description': 'Component-level Validation (21 checks)'
        },
        'test': {
            'script': validation_dir / 'comprehensive_test.py',
            'description': 'Integration Tests (9 tests)'
        },
        'report': {
            'script': validation_dir / 'diagnostic_report.py',
            'description': 'Diagnostic Report Generation'
        }
    }
    
    if validation_type and validation_type not in validations:
        print(f"✗ Unknown validation type: {validation_type}")
        print(f"Valid options: {', '.join(validations.keys())}")
        return 1
    
    print("\n" + "="*80)
    print("NOISE GENOME ESTIMATOR - VALIDATION FRAMEWORK")
    print("="*80 + "\n")
    
    results = {}
    
    if validation_type:
        # Run specific validation
        val = validations[validation_type]
        print(f"Running: {val['description']}")
        print("-" * 80)
        
        try:
            result = subprocess.run(
                [sys.executable, str(val['script'])],
                cwd=str(project_root),
                capture_output=False
            )
            results[validation_type] = result.returncode == 0
        except Exception as e:
            print(f"✗ Error: {e}")
            results[validation_type] = False
    else:
        # Run all validations
        for val_type, val_info in validations.items():
            print(f"Running: {val_info['description']}")
            print("-" * 80)
            
            try:
                result = subprocess.run(
                    [sys.executable, str(val_info['script'])],
                    cwd=str(project_root),
                    capture_output=False
                )
                results[val_type] = result.returncode == 0
                print()
            except Exception as e:
                print(f"✗ Error: {e}\n")
                results[val_type] = False
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for val_type, success in results.items():
        val_info = validations[val_type]
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {val_info['description']}")
    
    print(f"\nTotal: {passed}/{total} validations passed\n")
    
    if passed == total:
        print("✓ ALL VALIDATIONS PASSED!\n")
        print("Generated Files:")
        print("  - validation/FINAL_VALIDATION_REPORT.md")
        print("  - validation/DIAGNOSTIC_REPORT.md")
        print("  - validation/VALIDATION_SUMMARY.md")
        print("\nNext Steps:")
        print("  1. Review validation/FINAL_VALIDATION_REPORT.md")
        print("  2. Prepare datasets in ./datasets/{train,val,test}/")
        print("  3. Start training: python train.py --mode train")
        return 0
    else:
        print(f"✗ {total - passed} validation(s) failed\n")
        return 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Noise Genome Estimator validation framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_validation.py              # Run all validations
  python scripts/run_validation.py logic        # Component validation
  python scripts/run_validation.py test         # Integration tests
  python scripts/run_validation.py report       # Generate report
        """
    )
    
    parser.add_argument(
        'validation_type',
        nargs='?',
        default=None,
        choices=['logic', 'test', 'report'],
        help='Type of validation to run (default: run all)'
    )
    
    args = parser.parse_args()
    
    return run_validation(args.validation_type)


if __name__ == '__main__':
    sys.exit(main())
