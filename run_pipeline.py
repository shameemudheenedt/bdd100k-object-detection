#!/usr/bin/env python3
"""
Master script to run complete BDD100K pipeline
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n ERROR: {description} failed!")
        return False
    
    print(f"\n SUCCESS: {description} completed!")
    return True


def verify_setup():
    """Verify project setup."""
    print("\n Verifying setup...")
    return run_command("python test_setup.py", "Setup Verification")


def run_data_analysis():
    """Run data analysis."""
    print("\n Running data analysis...")
    return run_command("python data_analysis.py", "Data Analysis")


def prepare_dataset():
    """Prepare dataset for training."""
    print("\n Preparing dataset...")
    return run_command("python model_pipeline.py --task prepare", "Dataset Preparation")


def train_model(epochs=1):
    """Train model."""
    print(f"\n🤖 Training model ({epochs} epoch(s))...")
    cmd = f"python model_pipeline.py --task train --epochs {epochs}"
    return run_command(cmd, f"Model Training ({epochs} epochs)")


def run_inference(weights):
    """Run inference."""
    print("\n🔮 Running inference...")
    val_dir = "data/bdd100k_images_100k/bdd100k/images/100k/val"
    cmd = f"python inference.py --weights {weights} --source {val_dir} --output predictions.json"
    return run_command(cmd, "Inference")


def run_evaluation():
    """Run evaluation."""
    print("\n📈 Running evaluation...")
    return run_command("python run_evaluation.py", "Evaluation")


def launch_dashboard():
    """Launch Streamlit dashboard."""
    print("\n🎨 Launching dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    subprocess.run("streamlit run dashboard.py", shell=True)


def main():
    parser = argparse.ArgumentParser(description="BDD100K Pipeline Runner")
    parser.add_argument('--mode', choices=['all', 'analysis', 'model', 'eval', 'dashboard'],
                       default='all', help='Pipeline mode to run')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--weights', default='runs/train/bdd100k_exp/weights/best.pt',
                       help='Model weights for inference')
    parser.add_argument('--skip-verify', action='store_true', help='Skip setup verification')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         BDD100K Object Detection Pipeline                ║
    ║                                                          ║
    ║  Complete end-to-end pipeline for object detection      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Verify setup
    if not args.skip_verify:
        if not verify_setup():
            print("\n Setup verification failed. Please fix issues and try again.")
            return 1
    
    success = True
    
    # Run based on mode
    if args.mode in ['all', 'analysis']:
        success = success and run_data_analysis()
        
        if success:
            print("\n Analysis outputs saved to: analysis_output/")
            print("   - class_distribution.png")
            print("   - bbox_statistics.png")
            print("   - attributes_distribution.png")
            print("   - ANALYSIS_REPORT.md")
    
    if args.mode in ['all', 'model']:
        if success:
            success = success and prepare_dataset()
        
        if success:
            print("\n Dataset prepared at: bdd100k_yolo/")
            
            train_choice = input("\n  Training can take hours. Continue? (y/n): ")
            if train_choice.lower() == 'y':
                success = success and train_model(args.epochs)
                
                if success:
                    print(f"\n Model saved to: runs/train/bdd100k_exp/")
            else:
                print("\n⏭️  Skipping training. Use pre-trained weights for evaluation.")
    
    if args.mode in ['all', 'eval']:
        # Use pre-trained weights from repository
        pretrained_weights = "yolov5s_bdd100k/runs/exp0_yolov5s_bdd_prew/weights/best_yolov5s_bdd_prew.pt"
        
        if success:
            weights_path = Path(args.weights)
            
            # Check if custom weights exist, otherwise use pre-trained
            if not weights_path.exists():
                print(f"\n  Custom weights not found: {args.weights}")
                print(f"   Using pre-trained weights: {pretrained_weights}")
                args.weights = pretrained_weights
            
            weights_path = Path(args.weights)
            if weights_path.exists():
                success = success and run_evaluation()
                
                if success:
                    print("\n Evaluation outputs saved to:")
                    print("   - evaluation_output/")
            else:
                print(f"\n No weights available for evaluation")
                success = False
    
    if args.mode == 'dashboard':
        launch_dashboard()
        return 0
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n Generated Outputs:")
        print("   1. analysis_output/     - Data analysis results")
        print("   2. bdd100k_yolo/        - Prepared dataset")
        print("   3. runs/train/          - Training logs & weights")
        print("   4. predictions.json     - Model predictions")
        print("   5. evaluation_output/   - Evaluation metrics")
        print("   6. visualizations/      - Qualitative results")
        
        print("\n Next Steps:")
        print("   1. Review ANALYSIS_REPORT.md for insights")
        print("   2. Launch dashboard: python run_pipeline.py --mode dashboard")
        print("   3. Check evaluation_output/ for metrics")
        print("   4. Explore visualizations/ for failure cases")
        
        print("\n Documentation:")
        print("   - README.md              - Full documentation")
        print("   - QUICKSTART.md          - Quick start guide")
        print("   - MODEL_ARCHITECTURE.md  - Architecture details")
        print("   - PROJECT_SUMMARY.md     - Project summary")
        
    else:
        print(" PIPELINE FAILED!")
        print("="*60)
        print("\nPlease check error messages above and:")
        print("   1. Verify dataset paths")
        print("   2. Check dependencies: pip install -r requirements.txt")
        print("   3. Review logs for specific errors")
        print("   4. Run: python test_setup.py")
    
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
