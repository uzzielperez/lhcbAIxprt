#!/usr/bin/env python3
"""
Setup script for Hugging Face Vision Models
Installs dependencies and tests basic functionality
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Hugging Face vision requirements"""
    print("🔧 Installing Hugging Face Vision Dependencies...")
    
    requirements = [
        "transformers",
        "torch", 
        "torchvision",
        "accelerate",
        "bitsandbytes"  # For memory optimization
    ]
    
    for req in requirements:
        try:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    return True

def test_basic_imports():
    """Test if basic imports work"""
    print("\n🧪 Testing Basic Imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        from PIL import Image
        print("✅ Pillow")
        
        # Test if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available - will use CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_huggingface_processor():
    """Test the HuggingFace processor"""
    print("\n🤖 Testing HuggingFace Vision Processor...")
    
    try:
        from huggingface_vision import get_hf_processor
        
        processor = get_hf_processor()
        print("✅ HuggingFace processor created")
        
        # Test available models
        models = processor.get_available_models()
        print(f"✅ Available models: {len(models)}")
        
        for key, info in models.items():
            print(f"  - {info['name']} ({info['size']}) - {processor.estimate_memory_usage(key)}")
        
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace processor error: {e}")
        return False

def test_model_loading():
    """Test loading a lightweight model"""
    print("\n🔄 Testing Model Loading (Moondream 2B - Lightweight)...")
    
    try:
        from huggingface_vision import get_hf_processor
        
        processor = get_hf_processor()
        
        # Try to load the lightest model
        print("Loading Moondream 2B model...")
        success = processor.load_model("moondream2")
        
        if success:
            print("✅ Model loaded successfully!")
            
            # Test model info
            info = processor.get_model_info()
            print(f"Model info: {info}")
            
            # Unload to free memory
            processor.unload_model()
            print("✅ Model unloaded successfully")
            
            return True
        else:
            print("❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    """Main setup and test function"""
    print("🤗 HuggingFace Vision Models Setup & Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("shifter_rag_app_vision.py"):
        print("❌ Please run this script from the lhcbAIxprt directory")
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements")
        return False
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test processor
    if not test_huggingface_processor():
        print("\n❌ Processor tests failed") 
        return False
    
    # Test model loading (optional - requires memory)
    print("\n" + "="*50)
    print("🔄 Optional: Test Model Loading")
    response = input("Would you like to test loading a model? This requires 4-6GB RAM (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        if test_model_loading():
            print("✅ Model loading test passed!")
        else:
            print("⚠️ Model loading test failed - may need more memory")
    else:
        print("ℹ️ Skipping model loading test")
    
    print("\n" + "="*50)
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Run: streamlit run shifter_rag_app_vision.py")
    print("2. Go to the sidebar and expand '🔧 Model Management'")
    print("3. Click 'Load' for any model you want to use")
    print("4. Upload images in the Vision Analysis tab")
    
    print("\n💡 Model Recommendations:")
    print("- Start with 'Moondream 2B' (lightweight, ~4GB RAM)")
    print("- For better quality: 'LLaVA 1.5 7B' (~14GB RAM)")
    print("- For fastest: 'BLIP-2 OPT 2.7B' (~5GB RAM)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
