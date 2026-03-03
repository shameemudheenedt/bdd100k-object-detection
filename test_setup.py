"""
Test script to verify BDD100K analysis setup
"""
import json
from pathlib import Path
import sys


def check_dataset_structure():
    """Verify dataset files exist."""
    print("=" * 60)
    print("CHECKING DATASET STRUCTURE")
    print("=" * 60)
    
    base_dir = Path('/home/hp/Documents/Impliment/data')
    
    checks = {
        'Train Labels': base_dir / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json',
        'Val Labels': base_dir / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
        'Train Images': base_dir / 'bdd100k_images_100k/bdd100k/images/100k/train',
        'Val Images': base_dir / 'bdd100k_images_100k/bdd100k/images/100k/val'
    }
    
    all_good = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    return all_good


def check_label_format():
    """Verify label JSON format."""
    print("\n" + "=" * 60)
    print("CHECKING LABEL FORMAT")
    print("=" * 60)
    
    label_path = Path('/home/hp/Documents/Impliment/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json')
    
    if not label_path.exists():
        print("✗ Label file not found")
        return False
    
    try:
        with open(label_path) as f:
            data = json.load(f)
        
        print(f"✓ JSON loaded successfully")
        print(f"✓ Total images: {len(data)}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"✓ Sample image: {sample.get('name', 'N/A')}")
            print(f"✓ Has attributes: {'attributes' in sample}")
            print(f"✓ Has labels: {'labels' in sample}")
            
            if 'labels' in sample and len(sample['labels']) > 0:
                label = sample['labels'][0]
                print(f"✓ Sample label category: {label.get('category', 'N/A')}")
                print(f"✓ Has box2d: {'box2d' in label}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'cv2', 'PIL', 'torch', 'yaml', 'tqdm', 'streamlit'
    ]
    
    all_installed = True
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_yolo_repo():
    """Check YOLOv5 repository."""
    print("\n" + "=" * 60)
    print("CHECKING YOLOV5 REPOSITORY")
    print("=" * 60)
    
    yolo_dir = Path('/home/hp/Documents/Impliment/yolov5s_bdd100k')
    
    checks = {
        'Repository': yolo_dir,
        'Train script': yolo_dir / 'train.py',
        'Detect script': yolo_dir / 'detect.py',
        'Models': yolo_dir / 'models',
        'Weights': yolo_dir / 'weights/yolov5s.pt'
    }
    
    all_good = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    return all_good


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("BDD100K PROJECT SETUP VERIFICATION")
    print("=" * 60 + "\n")
    
    results = {
        'Dataset Structure': check_dataset_structure(),
        'Label Format': check_label_format(),
        'Dependencies': check_dependencies(),
        'YOLOv5 Repository': check_yolo_repo()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to run analysis!")
        print("\nNext steps:")
        print("1. Run: python data_analysis.py")
        print("2. Run: streamlit run dashboard.py")
        print("3. See QUICKSTART.md for full guide")
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above")
        print("\nCommon fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Verify dataset paths in scripts")
        print("- Check YOLOv5 repository is cloned")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
