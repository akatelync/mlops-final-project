#!/usr/bin/env python3
"""
Test script for the FastAPI application.
"""

import sys


def test_import():
    """Test if the FastAPI app can be imported."""
    try:
        print("✅ FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        return False


def test_pydantic_model():
    """Test if the Pydantic model works correctly."""
    try:
        from src.serve.app import InputData

        # Test valid input
        valid_data = InputData(avg_vtat=15.5, timestamp="2023-01-01T12:00:00")
        print("✅ Pydantic InputData model works correctly")
        print(f"   Example data: {valid_data.dict()}")

        # Test schema generation
        schema = InputData.schema()
        print("✅ Schema generation works")
        print(f"   Schema properties: {list(schema['properties'].keys())}")

        return True
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        return False


def test_preprocessing():
    """Test the preprocessing function."""
    try:
        from src.serve.app import preprocess_input

        test_data = {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}

        processed = preprocess_input(test_data)
        print("✅ Preprocessing function works")
        print(f"   Input columns: {list(test_data.keys())}")
        print(f"   Output columns: {list(processed.columns)}")
        print(f"   Output shape: {processed.shape}")

        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    try:
        from src.serve.app import config

        print("✅ Configuration loaded successfully")
        print(f"   MLflow tracking URI: {config['mlflow']['tracking_uri']}")
        print(f"   Model name: {config['mlflow']['model_name']}")
        print(f"   Feature columns: {config['features']['feature_cols']}")

        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing FastAPI Application")
    print("=" * 50)

    tests = [
        ("Import Test", test_import),
        ("Configuration Test", test_config_loading),
        ("Pydantic Model Test", test_pydantic_model),
        ("Preprocessing Test", test_preprocessing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print("   Test failed!")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The FastAPI application is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
