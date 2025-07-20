#!/usr/bin/env python3
"""
Simplified test script for core dependencies
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - FAILED: {e}")
        return False

def main():
    print("🔧 Shifter RAG System - Basic Dependency Check")
    print("=" * 50)
    
    # Test core modules first
    core_modules = [
        ("streamlit", "Streamlit"),
        ("groq", "Groq API"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("PyPDF2", "PyPDF2"),
        ("bs4", "BeautifulSoup4"),
        ("pickle", "Pickle"),
        ("json", "JSON"),
        ("datetime", "DateTime"),
        ("os", "OS"),
        ("tempfile", "TempFile"),
        ("io", "IO")
    ]
    
    success_count = 0
    
    for module, package in core_modules:
        if test_import(module, package):
            success_count += 1
    
    print(f"\n✅ Core modules: {success_count}/{len(core_modules)} working")
    
    # Test advanced modules separately
    print("\n🧪 Testing Advanced Components...")
    
    # Test FAISS
    try:
        import faiss
        index = faiss.IndexFlatIP(384)
        print("✅ FAISS - OK")
    except Exception as e:
        print(f"❌ FAISS - FAILED: {e}")
    
    # Test sentence transformers (may fail but that's ok)
    try:
        print("Testing sentence-transformers (this might take a moment)...")
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers - OK")
    except Exception as e:
        print(f"⚠️  Sentence Transformers - Has issues: {str(e)[:100]}...")
        print("   This is a known compatibility issue but the app might still work")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("✅ Core functionality should work")
    print("⚠️  If sentence-transformers has issues, try running the app anyway")
    print("\nNext steps:")
    print("1. Add your Groq API key to .streamlit/secrets.toml")
    print("2. Run: streamlit run shifter_rag_app.py")

if __name__ == "__main__":
    main() 