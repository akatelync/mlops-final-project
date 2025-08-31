#!/usr/bin/env python3
"""
Simple test runner for the MLOps project tests.

This script can be used to run tests without pytest if it's not installed.
For full pytest functionality, install pytest: pip install pytest
"""

import sys
import traceback
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def run_basic_import_tests():
    """Run basic import tests to verify modules can be loaded."""
    print("🧪 Running Basic Import Tests")
    print("=" * 50)

    tests_passed = 0
    tests_total = 0

    # Test 1: Import transform module
    tests_total += 1
    try:
        from src.features.transform import load_config

        print("✅ Transform module imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Transform module import failed: {e}")

    # Test 2: Import train module
    tests_total += 1
    try:
        from src.models.train import load_config

        print("✅ Train module imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Train module import failed: {e}")

    # Test 3: Import API module
    tests_total += 1
    try:
        from src.serve.app import InputData

        print("✅ API module imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ API module import failed: {e}")

    # Test 4: Test InputData model
    tests_total += 1
    try:
        from src.serve.app import InputData

        data = InputData(avg_vtat=15.5, timestamp="2023-01-01T12:00:00")
        assert data.avg_vtat == 15.5
        assert data.timestamp == "2023-01-01T12:00:00"
        print("✅ InputData model works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ InputData model test failed: {e}")

    # Test 5: Test config loading
    tests_total += 1
    try:
        from src.features.transform import load_config

        config = load_config("config.yaml")
        assert isinstance(config, dict)
        assert "features" in config
        assert "model" in config
        print("✅ Config loading works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        print("🎉 All basic tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


def run_transform_tests():
    """Run basic transformation tests."""
    print("\n🔄 Running Transform Tests")
    print("=" * 30)

    try:
        import pandas as pd

        from src.features.transform import transform_data

        # Create sample data
        sample_data = pd.DataFrame(
            {
                "avg_vtat": [10.5, 15.2, 8.7],
                "timestamp": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T14:30:00",
                    "2023-01-01T09:15:00",
                ],
                "is_cancelled": [0, 1, 0],
            }
        )

        # Test inference mode
        result = transform_data(
            input_df=sample_data, config_path="config.yaml", for_inference=True
        )

        # Basic assertions
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "avg_vtat" in result.columns
        assert "timestamp_hour" in result.columns

        print("✅ Transform inference test passed")
        return True

    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all available tests."""
    print("🧪 MLOps Project Test Suite")
    print("=" * 50)

    all_passed = True

    # Run basic import tests
    if not run_basic_import_tests():
        all_passed = False

    # Run transform tests
    if not run_transform_tests():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests completed successfully!")
        print("\n📝 To run the full test suite with pytest:")
        print("   pip install pytest")
        print("   python -m pytest tests/ -v")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
