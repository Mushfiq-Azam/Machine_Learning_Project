#!/usr/bin/env python3
"""
Life Satisfaction AI Assistant - Startup Script
This script checks dependencies and launches the Streamlit application.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'joblib',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        "life_satisfaction_model.pkl",
        "scaler.pkl",
        "label_encoder.pkl", 
        "X_columns.pkl"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required model files: {', '.join(missing_files)}")
        print("Please ensure all model files are in the current directory.")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🌱 Life Satisfaction AI Assistant")
    print("=" * 40)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install required packages")
        sys.exit(1)
    
    # Check model files
    print("Checking model files...")
    if not check_model_files():
        print("❌ Missing required model files")
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Starting Life Satisfaction AI Assistant...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 40)
    
    # Run Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "load_model_streamlit.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
