"""
Test script to verify project setup
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "matplotlib": "Matplotlib",
        "PyQt5": "PyQt5",
    }
    
    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0

def test_directory_structure():
    """Test that required directories exist"""
    print("\nTesting directory structure...")
    
    base_path = Path(__file__).parent
    required_dirs = [
        "data/Videos",
        "data/Annotations",
        "src",
        "models",
        "checkpoints",
        "logs",
        "outputs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_data_files():
    """Test that data files exist"""
    print("\nTesting data files...")
    
    base_path = Path(__file__).parent
    videos_dir = base_path / "data" / "Videos"
    annotations_dir = base_path / "data" / "Annotations"
    
    if not videos_dir.exists():
        print("  ✗ Videos directory not found")
        return False
    
    if not annotations_dir.exists():
        print("  ✗ Annotations directory not found")
        return False
    
    # Count files
    video_files = list(videos_dir.glob("*.mp4"))
    annotation_files = list(annotations_dir.glob("*.json"))
    
    print(f"  ✓ Found {len(video_files)} video files")
    print(f"  ✓ Found {len(annotation_files)} annotation files")
    
    if len(video_files) == 0 or len(annotation_files) == 0:
        print("  ⚠ Warning: Dataset appears to be incomplete")
        return False
    
    return True

def test_source_files():
    """Test that all source files exist"""
    print("\nTesting source files...")
    
    base_path = Path(__file__).parent
    src_dir = base_path / "src"
    
    required_files = [
        "config.py",
        "dataset.py",
        "model.py",
        "train.py",
        "evaluate.py",
        "decision_maker.py",
        "gui.py",
        "main.py",
        "__init__.py",
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = src_dir / file_name
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_name}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print("\nTesting PyTorch and CUDA...")
    
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    status = "✓" if cuda_available else "✗"
    print(f"  {status} CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ Warning: CUDA not available, will use CPU (slower training)")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Violence Detection System - Setup Verification")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Data Files", test_data_files),
        ("Source Files", test_source_files),
        ("PyTorch & CUDA", test_torch_cuda),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. cd src")
        print("  2. python main.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing data: Check data/Videos/ and data/Annotations/")
        print("  - Missing source files: Verify src/ directory contents")
        return 1

if __name__ == "__main__":
    sys.exit(main())
