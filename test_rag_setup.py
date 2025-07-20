#!/usr/bin/env python3
"""
Test script for Shifter RAG System setup verification
Run this script to check if all dependencies are installed correctly
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(module_name)
            print(f"✅ {package_name} - OK")
            return True
        else:
            importlib.import_module(module_name)
            print(f"✅ {module_name} - OK")
            return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - FAILED: {e}")
        return False

def main():
    print("🔧 Shifter RAG System - Dependency Check")
    print("=" * 50)
    
    required_modules = [
        ("streamlit", "Streamlit"),
        ("groq", "Groq API"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("PyPDF2", "PyPDF2"),
        ("bs4", "BeautifulSoup4"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("pathlib", "Pathlib"),
        ("pickle", "Pickle"),
        ("re", "Regular Expressions"),
        ("json", "JSON"),
        ("datetime", "DateTime"),
        ("typing", "Typing"),
        ("os", "OS"),
        ("tempfile", "TempFile"),
        ("io", "IO")
    ]
    
    success_count = 0
    total_count = len(required_modules)
    
    for module, package in required_modules:
        if test_import(module, package):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {success_count}/{total_count} dependencies available")
    
    if success_count == total_count:
        print("🎉 All dependencies are installed correctly!")
        print("\nNext steps:")
        print("1. Add your Groq API key to .streamlit/secrets.toml")
        print("2. Run: streamlit run shifter_rag_app.py")
    else:
        print("⚠️  Some dependencies are missing.")
        print("Run: pip install -r requirements_rag.txt")
    
    # Test basic functionality
    print("\n" + "=" * 50)
    print("🧪 Testing Core Components...")
    
    try:
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["test sentence"])
        print(f"✅ Embedding generation - OK (dimension: {len(test_embedding[0])})")
        
        # Test FAISS
        import faiss
        index = faiss.IndexFlatIP(384)
        print("✅ FAISS index creation - OK")
        
        # Test document processing
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<html><body>Test</body></html>", 'html.parser')
        print("✅ HTML parsing - OK")
        
        print("\n🎉 All core components working correctly!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
    
    print("\n" + "=" * 50)
    print("📋 Configuration Checklist:")
    print("□ Install dependencies: pip install -r requirements_rag.txt")
    print("□ Get Groq API key: https://console.groq.com/keys")
    print("□ Configure secrets: Edit .streamlit/secrets.toml")
    print("□ Run application: streamlit run shifter_rag_app.py")

if __name__ == "__main__":
    main() 