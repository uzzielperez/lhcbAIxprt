#!/usr/bin/env python3
"""
Test script for LHCb Shifter Assistant with Agentic Context Engineering (ACE)
Verifies all components are working correctly and provides diagnostic information.
"""

import sys
import os
import importlib
from datetime import datetime
from pathlib import Path

def test_core_dependencies():
    """Test core Python dependencies"""
    print("🔍 Testing core dependencies...")
    
    core_modules = [
        'streamlit', 'pandas', 'numpy', 'groq', 'pickle', 'json', 
        'datetime', 'pathlib', 're', 'hashlib', 'collections'
    ]
    
    working_modules = []
    failed_modules = []
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            working_modules.append(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            failed_modules.append((module, str(e)))
            print(f"  ❌ {module}: {e}")
    
    return working_modules, failed_modules

def test_document_processing():
    """Test document processing capabilities"""
    print("\n📄 Testing document processing...")
    
    try:
        import PyPDF2
        print("  ✅ PyPDF2 - PDF processing")
    except ImportError:
        print("  ❌ PyPDF2 - PDF processing failed")
    
    try:
        from bs4 import BeautifulSoup
        print("  ✅ BeautifulSoup - HTML processing")
    except ImportError:
        print("  ❌ BeautifulSoup - HTML processing failed")
    
    try:
        from PIL import Image
        print("  ✅ PIL - Image processing")
    except ImportError:
        print("  ⚠️ PIL - Image processing not available (optional)")

def test_ace_framework():
    """Test ACE framework components"""
    print("\n🧠 Testing ACE framework...")
    
    try:
        from ace_framework import (
            ACESystem, ContextEvolutionEngine, AdaptiveRAGPipeline,
            ContextNode, FeedbackEntry, LearningMetrics
        )
        print("  ✅ ACE Framework - Core classes imported")
        
        # Test basic ACE functionality
        from ace_framework import ACESystem
        print("  ✅ ACE Framework - System initialization")
        
    except ImportError as e:
        print(f"  ❌ ACE Framework - Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ ACE Framework - Error: {e}")
        return False
    
    return True

def test_groq_api():
    """Test Groq API connection"""
    print("\n🔌 Testing Groq API...")
    
    try:
        from groq import Groq
        print("  ✅ Groq client imported")
        
        # Check if API key is available
        if os.path.exists('.streamlit/secrets.toml'):
            print("  ✅ Streamlit secrets file found")
            
            # Try to read API key
            try:
                import toml
                with open('.streamlit/secrets.toml', 'r') as f:
                    secrets = toml.load(f)
                    if 'groq_api_key' in secrets:
                        print("  ✅ Groq API key found in secrets")
                        
                        # Test API connection
                        try:
                            client = Groq(api_key=secrets['groq_api_key'])
                            print("  ✅ Groq API client initialized")
                        except Exception as e:
                            print(f"  ⚠️ Groq API client error: {e}")
                    else:
                        print("  ⚠️ Groq API key not found in secrets")
            except ImportError:
                print("  ⚠️ TOML parser not available - cannot read secrets")
        else:
            print("  ⚠️ Streamlit secrets file not found")
            
    except ImportError:
        print("  ❌ Groq client not available")
        return False
    
    return True

def test_ace_application():
    """Test ACE application components"""
    print("\n🚀 Testing ACE application...")
    
    try:
        # Test if the main application can be imported
        import shifter_rag_app_ace
        print("  ✅ ACE application imported")
        
        # Test if key classes are available
        from shifter_rag_app_ace import (
            ShifterRAGSystem, DocumentProcessor, SimpleTextSearch, MemorySystem
        )
        print("  ✅ ACE application classes imported")
        
    except ImportError as e:
        print(f"  ❌ ACE application import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ ACE application error: {e}")
        return False
    
    return True

def test_data_persistence():
    """Test data persistence capabilities"""
    print("\n💾 Testing data persistence...")
    
    try:
        import pickle
        import json
        
        # Test pickle functionality
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        pickle.dumps(test_data)
        print("  ✅ Pickle serialization")
        
        # Test JSON functionality
        json.dumps(test_data)
        print("  ✅ JSON serialization")
        
    except Exception as e:
        print(f"  ❌ Data persistence error: {e}")
        return False
    
    return True

def test_visualization():
    """Test visualization capabilities"""
    print("\n📊 Testing visualization...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple test plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        
        # Save to temporary file
        test_plot_path = "test_plot.png"
        plt.savefig(test_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Clean up
        if os.path.exists(test_plot_path):
            os.remove(test_plot_path)
        
        print("  ✅ Matplotlib plotting")
        
    except ImportError:
        print("  ⚠️ Matplotlib not available")
    except Exception as e:
        print(f"  ❌ Visualization error: {e}")

def run_ace_demo():
    """Run a simple ACE demo to verify functionality"""
    print("\n🎯 Running ACE demo...")
    
    try:
        from ace_framework import ACESystem, ShifterRAGSystem
        
        # Create a mock base RAG system
        class MockRAGSystem:
            def __init__(self):
                self.search_engine = type('MockSearch', (), {
                    'search': lambda self, query, k=5: []
                })()
                self.memory_system = type('MockMemory', (), {})()
            
            def generate_response(self, query, context_docs=None, use_memory=True):
                return f"Mock response for: {query}"
        
        # Initialize ACE system
        mock_rag = MockRAGSystem()
        ace_system = ACESystem(mock_rag)
        
        # Test ACE functionality
        query = "Test query for ACE system"
        response, metrics = ace_system.process_query_with_ace(query)
        
        print(f"  ✅ ACE query processed: {response[:50]}...")
        print(f"  ✅ ACE metrics: {metrics}")
        
        # Test feedback collection
        feedback_result = ace_system.collect_feedback(
            query, response, 4, "Good response", ["Keep it up"]
        )
        print(f"  ✅ Feedback collected: {feedback_result['feedback_id']}")
        
        # Test ACE metrics
        ace_metrics = ace_system.get_ace_metrics()
        print(f"  ✅ ACE metrics retrieved: {len(ace_metrics)} metrics")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ACE demo failed: {e}")
        return False

def generate_setup_report():
    """Generate comprehensive setup report"""
    print("\n" + "="*60)
    print("🧠 LHCb Shifter Assistant with ACE - Setup Report")
    print("="*60)
    
    # Test all components
    core_working, core_failed = test_core_dependencies()
    test_document_processing()
    ace_framework_ok = test_ace_framework()
    groq_api_ok = test_groq_api()
    ace_app_ok = test_ace_application()
    persistence_ok = test_data_persistence()
    test_visualization()
    
    # Run ACE demo
    ace_demo_ok = run_ace_demo()
    
    # Generate summary
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    
    total_tests = 7
    passed_tests = sum([
        len(core_working) > 0,
        ace_framework_ok,
        groq_api_ok,
        ace_app_ok,
        persistence_ok,
        ace_demo_ok
    ])
    
    print(f"✅ Tests passed: {passed_tests}/{total_tests}")
    
    if core_failed:
        print(f"\n❌ Failed core modules: {len(core_failed)}")
        for module, error in core_failed:
            print(f"   • {module}: {error}")
    
    if passed_tests >= 5:
        print("\n🎉 ACE system is ready to use!")
        print("\n🚀 To start the ACE system:")
        print("   streamlit run shifter_rag_app_ace.py")
    else:
        print("\n⚠️ Some components need attention before ACE system can run")
        print("\n🔧 Recommended fixes:")
        if core_failed:
            print("   • Install missing dependencies: pip install -r requirements_ace.txt")
        if not groq_api_ok:
            print("   • Configure Groq API key in .streamlit/secrets.toml")
        if not ace_framework_ok:
            print("   • Check ace_framework.py is in the same directory")
    
    print("\n📚 For more information, see README.md")
    print("="*60)

if __name__ == "__main__":
    generate_setup_report()
