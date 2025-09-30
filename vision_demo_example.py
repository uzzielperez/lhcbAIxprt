#!/usr/bin/env python3
"""
Vision RAG Demo Script
Demonstrates how to use the vision capabilities programmatically
"""

import os
import sys
import base64
from PIL import Image
import io
from groq import Groq

def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple diagram-like image
    img = Image.new('RGB', (400, 300), color='white')
    
    # You would normally load a real image like this:
    # img = Image.open('path/to/your/image.jpg')
    
    return img

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64

def analyze_image_with_vision_llm(image_base64, query, api_key):
    """Analyze image using Groq's vision model"""
    try:
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert shifter assistant with vision capabilities. 
                    Analyze images of equipment, error screens, diagrams, and provide clear, 
                    actionable guidance for operations personnel."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {e}"

def main():
    """Demo the vision capabilities"""
    print("🔧👁️ LHCb AI Expert - Vision RAG Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        print("Set it with: export GROQ_API_KEY='your_api_key_here'")
        print("Or configure it in .streamlit/secrets.toml for the web app")
        return
    
    print("✅ API key found")
    
    # Create or load sample image
    print("🖼️  Creating sample image...")
    sample_image = create_sample_image()
    
    # In a real scenario, you would load an actual equipment photo:
    # sample_image = Image.open('equipment_photo.jpg')
    
    # Encode image
    print("🔄 Encoding image...")
    image_base64 = encode_image_to_base64(sample_image)
    
    # Example queries for different use cases
    demo_queries = [
        "Describe what you see in this image and identify any potential issues.",
        "What equipment or systems are visible in this image?",
        "If this were an error screen, what would you recommend checking?",
        "Analyze this image for any safety concerns or operational issues."
    ]
    
    print("\n🤖 Running Vision Analysis Demo:")
    print("-" * 40)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("   Response: ", end="")
        
        response = analyze_image_with_vision_llm(image_base64, query, api_key)
        print(response[:200] + "..." if len(response) > 200 else response)
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\nTo use the full web interface:")
    print("1. Run: streamlit run shifter_rag_app_vision.py")
    print("2. Go to the 'Vision Analysis' tab")
    print("3. Upload real equipment photos, error screens, or diagrams")
    print("4. Ask specific questions about what you see")

if __name__ == "__main__":
    main()
