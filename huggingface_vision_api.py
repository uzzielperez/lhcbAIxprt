#!/usr/bin/env python3
"""
Hugging Face Vision Models API Integration (Fixed)
Provides cloud-based vision model capabilities using Hugging Face Inference API
"""

import streamlit as st
import requests
import base64
import io
from PIL import Image
from typing import Dict, Any, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceVisionAPI:
    """Handle Hugging Face vision models via API"""
    
    def __init__(self):
        self.api_base_url = "https://api-inference.huggingface.co/models"
        self.api_key = None
        
        # Available models (ordered by preference/capability)
        self.available_models = {
            # "Salesforce/blip-image-captioning-base": {
            #     "name": "BLIP Image Captioning (Base)",
            #     "type": "image-to-text",
            #     "description": "Fast image captioning model",
            #     "max_size": 1024,
            #     "free_tier": True
            # },
            # "Salesforce/blip-vqa-base": {
            #     "name": "BLIP VQA (Base)",
            #     "type": "image-to-text", 
            #     "description": "Visual question answering",
            #     "max_size": 1024,
            #     "free_tier": True
            # },
            "microsoft/git-base": {
                "name": "GIT Base (Microsoft)",
                "type": "image-to-text",
                "description": "Good general-purpose image captioning",
                "max_size": 1024,
                "free_tier": True
            },
            "nlpconnect/vit-gpt2-image-captioning": {
                "name": "ViT-GPT2 Image Captioning",
                "type": "image-to-text",
                "description": "Lightweight image captioning model",
                "max_size": 1024,
                "free_tier": True
            },
            "ydshieh/vit-gpt2-coco-en": {
                "name": "ViT-GPT2 COCO (English)",
                "type": "image-to-text",
                "description": "Alternative lightweight captioning model",
                "max_size": 1024,
                "free_tier": True
            },
            # "Salesforce/blip2-opt-2.7b": {
            #     "name": "BLIP-2 OPT 2.7B (Paid)",
            #     "type": "image-to-text",
            #     "description": "Advanced vision-language model (requires paid API)",
            #     "max_size": 1024,
            #     "free_tier": False
            # }
        }
    
    def set_api_key(self, api_key: str):
        """Set Hugging Face API key"""
        self.api_key = api_key
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models"""
        return self.available_models
    
    def estimate_cost(self, model_key: str) -> str:
        """Estimate cost for model usage"""
        model_info = self.available_models.get(model_key, {})
        
        if model_info.get("free_tier", False):
            return "Free tier available (with rate limits)"
        else:
            return "Paid tier required"
    
    def analyze_image(self, image_data: Dict[str, Any], query: str, model_key: str = None) -> str:
        """Analyze image using Hugging Face API"""
        try:
            if not self.api_key:
                return "❌ Hugging Face API key not configured. Please add your API key in the sidebar."
            
            # Use default model if none specified
            if not model_key:
                model_key = "nlpconnect/vit-gpt2-image-captioning"
            
            if model_key not in self.available_models:
                return f"❌ Model {model_key} not found in available models."
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data['base64'])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Resize image if too large
            model_info = self.available_models[model_key]
            max_size = model_info.get("max_size", 1024)
            
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert image to bytes again
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            
            # Try the requested model first, then fall back to known working ones on 404
            fallback_models = [
                "nlpconnect/vit-gpt2-image-captioning",
                "ydshieh/vit-gpt2-coco-en",
                "microsoft/git-base"
            ]
            # Ensure we only try models we know about and avoid duplicates
            candidate_models = []
            if model_key in self.available_models:
                candidate_models.append(model_key)
            for m in fallback_models:
                if m in self.available_models and m not in candidate_models:
                    candidate_models.append(m)

            last_error_text = None
            last_status_code = None
            used_model_key = None

            for candidate_key in candidate_models:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                model_url = f"{self.api_base_url}/{candidate_key}"

                # Try 1: POST to /models with raw bytes payload (some endpoints accept direct image bytes)
                try:
                    headers_bytes = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "image/jpeg"}
                    response = requests.post(model_url, headers=headers_bytes, data=image_bytes, timeout=60)
                except Exception:
                    response = None

                # If not OK, Try 1b: multipart form data
                if not response or response.status_code != 200:
                    files = {
                        'inputs': ('image.jpg', image_bytes, 'image/jpeg')
                    }
                    if "vqa" in candidate_key.lower() and query.strip():
                        data = {
                            'parameters': f'{{"question": "{query}"}}'
                        }
                        response = requests.post(model_url, headers=headers, files=files, data=data, timeout=60)
                    else:
                        response = requests.post(model_url, headers=headers, files=files, timeout=60)

                if response.status_code == 200:
                    used_model_key = candidate_key
                    result = response.json()
                    break
                elif response.status_code == 404:
                    # Try 2: Same /models endpoint but JSON with base64 data URL
                    try:
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        data_url = f"data:image/jpeg;base64,{image_b64}"
                        if "vqa" in candidate_key.lower() and query.strip():
                            payload = {"inputs": {"image": data_url, "question": query}}
                        else:
                            payload = {"inputs": data_url}
                        headers_json = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                        response_json = requests.post(model_url, headers=headers_json, json=payload, timeout=60)
                        if response_json.status_code == 200:
                            used_model_key = candidate_key
                            result = response_json.json()
                            break
                        elif response_json.status_code != 404:
                            last_error_text = response_json.text
                            last_status_code = response_json.status_code
                            continue
                    except Exception:
                        pass

                    # Try 3: Fall back to /pipeline endpoint
                    pipeline_base = self.api_base_url.replace("/models", "/pipeline")
                    task = "visual-question-answering" if "vqa" in candidate_key.lower() else "image-to-text"
                    pipeline_url = f"{pipeline_base}/{task}/{candidate_key}"
                    try:
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        data_url = f"data:image/jpeg;base64,{image_b64}"
                        if task == "visual-question-answering" and query.strip():
                            payload = {"inputs": {"image": data_url, "question": query}}
                        else:
                            payload = {"inputs": data_url}
                        headers_json = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                        response_pipeline = requests.post(pipeline_url, headers=headers_json, json=payload, timeout=60)
                        if response_pipeline.status_code == 200:
                            used_model_key = candidate_key
                            result = response_pipeline.json()
                            break
                        else:
                            last_error_text = response_pipeline.text
                            last_status_code = response_pipeline.status_code
                            continue
                    except Exception as e:
                        last_error_text = str(e)
                        last_status_code = None
                        continue
                elif response.status_code in (503, 429, 401):
                    # Return immediately for loading/rate/auth errors since retrying another model may not help
                    if response.status_code == 503:
                        return f"""⚠️ **Model Loading**
                
The model {candidate_key} is currently loading. This can take a few minutes for the first request.

Please wait a moment and try again, or try a different model."""
                    if response.status_code == 429:
                        return f"""⚠️ **Rate Limit Exceeded**
                
You've hit the rate limit for the Hugging Face API. Please wait a moment before trying again.

Consider upgrading to a paid plan for higher rate limits."""
                    if response.status_code == 401:
                        return f"""❌ **Authentication Error**
                
Invalid API key. Please check your Hugging Face API key in the sidebar."""
                else:
                    last_error_text = response.text
                    last_status_code = response.status_code
                    continue

            # If we have a successful result, format and return it
            if used_model_key is not None:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        # BLIP response format
                        if "answer" in result[0]:
                            analysis = result[0]["answer"]
                        elif "generated_text" in result[0]:
                            analysis = result[0]["generated_text"]
                        else:
                            analysis = str(result[0])
                    else:
                        analysis = str(result[0])
                elif isinstance(result, dict):
                    if "generated_text" in result:
                        analysis = result["generated_text"]
                    else:
                        analysis = str(result)
                else:
                    analysis = str(result)
                
                # Enhance the response with context
                model_info = self.available_models[used_model_key]
                model_name = model_info["name"]
                enhanced_response = f"""🤖 **{model_name} Analysis:**

{analysis}

---
*Analysis provided by Hugging Face API using `{used_model_key}`*"""
                
                return enhanced_response

            # If we got here, all candidates failed
            tried_models = ", ".join(candidate_models)
            if last_status_code == 404:
                return f"""❌ **API Error (404 - Not Found)**
                
None of the endpoints for the tried models were found.
Tried models: {tried_models}

Consider selecting a different model or checking Hugging Face model availability."""
            else:
                return f"""❌ **API Error**
                
Status Code: {last_status_code}
Response: {str(last_error_text)[:300]}..."""
                
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The model might be taking too long to respond. Try a different model."
            
        except requests.exceptions.RequestException as e:
            return f"❌ Network error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"❌ Error analyzing image: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about API status"""
        return {
            "status": "API Ready" if self.api_key else "API Key Required",
            "api_configured": bool(self.api_key),
            "available_models": len(self.available_models),
            "free_models": len([m for m in self.available_models.values() if m.get("free_tier", False)])
        }
    
    def is_available(self) -> bool:
        """Check if Hugging Face API is available"""
        return True
    
    def test_api_connection(self) -> str:
        """Test API connection with a simple request"""
        try:
            if not self.api_key:
                return "❌ No API key configured"
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            test_url = f"{self.api_base_url}/nlpconnect/vit-gpt2-image-captioning"
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return "✅ API connection successful"
            elif response.status_code == 401:
                return "❌ Invalid API key"
            else:
                return f"⚠️ API responded with status {response.status_code}"
                
        except Exception as e:
            return f"❌ Connection test failed: {str(e)}"


# Singleton instance for the app
_hf_api = None

def get_hf_processor() -> HuggingFaceVisionAPI:
    """Get singleton Hugging Face API processor instance"""
    global _hf_api
    if _hf_api is None:
        _hf_api = HuggingFaceVisionAPI()
    return _hf_api