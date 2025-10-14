#!/usr/bin/env python3
"""
Test script for Hugging Face API integration
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from huggingface_vision_api import get_hf_processor
    print("✅ Successfully imported HuggingFace API processor")
    
    # Test basic functionality
    hf_processor = get_hf_processor()
    print("✅ Successfully created processor instance")
    
    # Test model listing
    models = hf_processor.get_available_models()
    print(f"✅ Found {len(models)} available models:")
    for key, info in models.items():
        print(f"  - {info['name']} ({info['type']})")
    
    # Test API info
    api_info = hf_processor.get_model_info()
    print(f"✅ API Status: {api_info['status']}")
    
    print("\n🎉 All tests passed! The HuggingFace API integration is working correctly.")
    print("\nTo use the API:")
    print("1. Get a free API key from https://huggingface.co/settings/tokens")
    print("2. Add it to the Streamlit app in the sidebar")
    print("3. Start analyzing images!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all files are present.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

