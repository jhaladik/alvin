#!/usr/bin/env python3
"""
NGC Setup Verification Script
Run this BEFORE training to verify everything is configured correctly
"""
import sys
import os

def check(description, test_func):
    """Run a check and print result"""
    sys.stdout.write(f"Checking {description}... ")
    sys.stdout.flush()

    try:
        result = test_func()
        if result:
            print("✓")
            return True
        else:
            print("✗")
            return False
    except Exception as e:
        print(f"✗ ({e})")
        return False

def main():
    print("=" * 70)
    print("NGC Setup Verification")
    print("=" * 70)
    print()

    all_good = True

    # Check 1: Python version
    def check_python():
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"  Python {version.major}.{version.minor}.{version.micro}")
            return True
        return False

    if not check("Python version (>=3.8)", check_python):
        print("  ⚠ Need Python 3.8+")
        all_good = False

    # Check 2: PyTorch
    def check_pytorch():
        import torch
        print(f"  PyTorch {torch.__version__}")
        return True

    if not check("PyTorch installed", check_pytorch):
        print("  ⚠ Run: pip install torch")
        all_good = False

    # Check 3: CUDA/GPU
    def check_cuda():
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  {gpu_name}")
            return True
        return False

    if not check("GPU/CUDA available", check_cuda):
        print("  ⚠ No GPU detected - training will be VERY slow!")
        print("  Make sure you selected GPU when creating NGC workspace")
        all_good = False

    # Check 4: Required packages
    packages = [
        'numpy',
        'requests',
        'onnx',
        'onnxruntime',
        'tqdm'
    ]

    for package in packages:
        def check_package(pkg=package):
            __import__(pkg)
            return True

        if not check(f"{package} installed", check_package):
            print(f"  ⚠ Run: pip install {package}")
            all_good = False

    # Check 5: Directory structure
    dirs = [
        'data',
        'preprocessing',
        'models',
        'training',
        'export'
    ]

    for directory in dirs:
        def check_dir(d=directory):
            return os.path.isdir(d)

        if not check(f"Directory '{directory}' exists", check_dir):
            print(f"  ⚠ Missing directory: {directory}")
            all_good = False

    # Check 6: Required files
    files = [
        'preprocessing/feature_extraction.py',
        'models/behavioral_cloning.py',
        'training/train.py',
        'export/to_onnx.py'
    ]

    for file in files:
        def check_file(f=file):
            return os.path.isfile(f)

        if not check(f"File '{file}' exists", check_file):
            print(f"  ⚠ Missing file: {file}")
            all_good = False

    # Check 7: Can connect to Cloudflare
    def check_cloudflare():
        import requests
        url = "https://alvin-pacman-ai.jhaladik.workers.dev/api/export-training-data"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            print(f"  {count} samples available")
            return count > 0
        return False

    if not check("Cloudflare Worker accessible", check_cloudflare):
        print("  ⚠ Can't reach worker - check URL and deployment")
        all_good = False

    # Check 8: Disk space
    def check_disk():
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)
        print(f"  {free_gb:.1f} GB free")
        return free_gb > 5

    if not check("Disk space (>5GB)", check_disk):
        print("  ⚠ Low disk space - may fail during training")
        all_good = False

    # Check 9: Memory
    def check_memory():
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024**2)
                        print(f"  {mem_gb:.1f} GB available")
                        return mem_gb > 8
        except:
            return True  # Can't check on all systems

    if not check("RAM (>8GB)", check_memory):
        print("  ⚠ Low memory - may need to reduce batch size")

    print()
    print("=" * 70)

    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("You're ready to train! Run:")
        print("  python quickstart.py")
        print()
        print("Or run steps manually:")
        print("  python data/fetch_training_data.py")
        print("  python preprocessing/feature_extraction.py")
        print("  python training/train.py")
        print("  python export/to_onnx.py")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Fix the issues above before training.")
        print("See NGC_SETUP_GUIDE.md for help.")
        return 1

if __name__ == "__main__":
    exit(main())
