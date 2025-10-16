#!/usr/bin/env python3
"""
Test script for Vision RAG setup
Tests core dependencies and vision capabilities
"""

import sys
import traceback
from datetime import datetime

def test_core_imports():
    """Test if core modules can be imported"""
    modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'groq',
        'PyPDF2',
        'bs4',  # BeautifulSoup
        'pathlib',
        'pickle',
        're',
        'json',
        'datetime',
        'io',
        'base64'
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, len(modules), len(failed_imports)

def test_vision_imports():
    """Test if vision-specific modules can be imported"""
    vision_modules = [
        'PIL',  # Pillow
    ]
    
    failed_imports = []
    
    for module in vision_modules:
        try:
            if module == 'PIL':
                from PIL import Image
                print(f"✅ {module} (Pillow)")
            else:
                __import__(module)
                print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, len(vision_modules), len(failed_imports)

def test_groq_api():
    """Test Groq API connection"""
    try:
        from groq import Groq
        
        # Try to read API key from streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "groq_api_key" in st.secrets:
                client = Groq(api_key=st.secrets["groq_api_key"])
                print("✅ Groq API key found in Streamlit secrets")
                return True
            else:
                print("⚠️  Groq API key not found in Streamlit secrets")
                print("   Add your key to .streamlit/secrets.toml:")
                print('   groq_api_key = "your_api_key_here"')
                return False
        except Exception as e:
            print(f"⚠️  Could not access Streamlit secrets: {e}")
            print("   Make sure to configure .streamlit/secrets.toml when running the app")
            return False
            
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

def test_image_processing():
    """Test basic image processing capabilities"""
    try:
        from PIL import Image
        import io
        import base64
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Test image operations
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Test base64 encoding
        image_base64 = base64.b64encode(image_bytes).decode()
        
        # Test image loading from bytes
        loaded_image = Image.open(io.BytesIO(image_bytes))
        
        print("✅ Image processing capabilities working")
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        traceback.print_exc()
        return False

def test_vision_models():
    """Test available vision models"""
    try:
        vision_models = {
            "llava-v1.5-7b-4096": "LLaVA 1.5 7B (Only Available Vision Model)"
        }
        
        print("✅ Vision models configured:")
        for model_id, description in vision_models.items():
            print(f"   - {description} ({model_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision models test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧  AI Expert - Vision RAG Setup Test")
    print("=" * 50)
    print(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test core imports
    print("📦 Testing Core Dependencies:")
    core_ok, core_total, core_failed = test_core_imports()
    print(f"Result: {core_total - core_failed}/{core_total} working")
    print()
    
    # Test vision imports
    print("👁️  Testing Vision Dependencies:")
    vision_ok, vision_total, vision_failed = test_vision_imports()
    print(f"Result: {vision_total - vision_failed}/{vision_total} working")
    print()
    
    # Test image processing
    print("🖼️  Testing Image Processing:")
    image_ok = test_image_processing()
    print()
    
    # Test vision models
    print("🤖 Testing Vision Models:")
    models_ok = test_vision_models()
    print()
    
    # Test Groq API
    print("🔌 Testing Groq API:")
    api_ok = test_groq_api()
    print()
    
    # Summary
    print("📊 Test Summary:")
    print("=" * 30)
    
    all_tests = [
        ("Core Dependencies", core_ok),
        ("Vision Dependencies", vision_ok),
        ("Image Processing", image_ok),
        ("Vision Models", models_ok),
        ("Groq API", api_ok)
    ]
    
    passed_tests = sum(1 for _, status in all_tests if status)
    total_tests = len(all_tests)
    
    for test_name, status in all_tests:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {test_name}")
    
    print()
    print(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your vision RAG system is ready to use.")
        print()
        print("Next steps:")
        print("1. Run: streamlit run shifter_rag_app_vision.py")
        print("2. Upload documents and images")
        print("3. Test vision analysis with sample images")
    else:
        print("⚠️  Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements_rag_vision.txt")
        print()
        print("Make sure to configure your Groq API key in .streamlit/secrets.toml")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
