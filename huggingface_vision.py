#!/usr/bin/env python3
"""
Hugging Face Vision Models Integration
Provides local vision model capabilities as alternative to Groq
"""

import streamlit as st
import torch
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceVisionProcessor:
    """Handle Hugging Face vision models for local processing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_processor = None
        self.current_model_name = None
        
        # Available models (ordered by preference/capability)
        self.available_models = {
            "moondream2": {
                "name": "Moondream 2B (Lightweight)",
                "model_id": "vikhyatk/moondream2",
                "type": "moondream",
                "size": "2B",
                "description": "Fast, lightweight vision-language model"
            },
            "llava-1.5-7b": {
                "name": "LLaVA 1.5 7B (Recommended)",
                "model_id": "llava-hf/llava-1.5-7b-hf",
                "type": "llava",
                "size": "7B", 
                "description": "High-quality vision-language model"
            },
            "llava-1.5-13b": {
                "name": "LLaVA 1.5 13B (High Quality)",
                "model_id": "llava-hf/llava-1.5-13b-hf", 
                "type": "llava",
                "size": "13B",
                "description": "Highest quality, requires more memory"
            },
            "blip2-opt-2.7b": {
                "name": "BLIP-2 OPT 2.7B (Fast)",
                "model_id": "Salesforce/blip2-opt-2.7b",
                "type": "blip2",
                "size": "2.7B",
                "description": "Fast image captioning and QA"
            }
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models"""
        return self.available_models
    
    def estimate_memory_usage(self, model_key: str) -> str:
        """Estimate memory usage for model"""
        model_info = self.available_models.get(model_key, {})
        size = model_info.get("size", "Unknown")
        
        memory_estimates = {
            "2B": "~4-6 GB RAM",
            "2.7B": "~5-7 GB RAM", 
            "7B": "~14-16 GB RAM",
            "13B": "~26-30 GB RAM"
        }
        
        return memory_estimates.get(size, "Unknown memory requirement")
    
    def load_model(self, model_key: str) -> bool:
        """Load specified Hugging Face model"""
        try:
            if model_key not in self.available_models:
                logger.error(f"Model {model_key} not found in available models")
                return False
            
            model_info = self.available_models[model_key]
            model_id = model_info["model_id"]
            model_type = model_info["type"]
            
            logger.info(f"Loading {model_info['name']} ({model_id})")
            
            # Clear previous model from memory
            if self.current_model is not None:
                del self.current_model
                del self.current_processor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load model based on type
            if model_type == "moondream":
                self._load_moondream_model(model_id)
            elif model_type == "llava":
                self._load_llava_model(model_id)
            elif model_type == "blip2":
                self._load_blip2_model(model_id)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False
            
            self.current_model_name = model_key
            logger.info(f"Successfully loaded {model_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {str(e)}")
            return False
    
    def _load_moondream_model(self, model_id: str):
        """Load Moondream model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.current_processor = AutoTokenizer.from_pretrained(model_id)
        
        if not torch.cuda.is_available():
            self.current_model = self.current_model.to(self.device)
    
    def _load_llava_model(self, model_id: str):
        """Load LLaVA model"""
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        self.current_processor = LlavaNextProcessor.from_pretrained(model_id)
        self.current_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            self.current_model = self.current_model.to(self.device)
    
    def _load_blip2_model(self, model_id: str):
        """Load BLIP-2 model"""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.current_processor = Blip2Processor.from_pretrained(model_id)
        self.current_model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            self.current_model = self.current_model.to(self.device)
    
    def analyze_image(self, image_data: Dict[str, Any], query: str) -> str:
        """Analyze image using loaded Hugging Face model"""
        try:
            if self.current_model is None:
                return "❌ No model loaded. Please select and load a model first."
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data['base64'])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            model_info = self.available_models[self.current_model_name]
            model_type = model_info["type"]
            
            # Generate response based on model type
            if model_type == "moondream":
                response = self._analyze_with_moondream(image, query)
            elif model_type == "llava":
                response = self._analyze_with_llava(image, query)
            elif model_type == "blip2":
                response = self._analyze_with_blip2(image, query)
            else:
                return f"❌ Unsupported model type: {model_type}"
            
            return f"🤖 **{model_info['name']} Analysis:**\n\n{response}"
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"❌ Error analyzing image: {str(e)}"
    
    def _analyze_with_moondream(self, image: Image.Image, query: str) -> str:
        """Analyze with Moondream model"""
        if not query.strip():
            query = "Describe this image in detail, focusing on any technical aspects, measurements, or important information visible."
        
        response = self.current_model.answer_question(image, query, self.current_processor)
        return response
    
    def _analyze_with_llava(self, image: Image.Image, query: str) -> str:
        """Analyze with LLaVA model"""
        if not query.strip():
            query = "Describe this image in detail, focusing on any technical aspects, measurements, or important information visible."
        
        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt = self.current_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.current_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.current_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
        
        response = self.current_processor.decode(output[0], skip_special_tokens=True)
        # Extract just the assistant's response
        response = response.split("assistant\n")[-1] if "assistant\n" in response else response
        
        return response.strip()
    
    def _analyze_with_blip2(self, image: Image.Image, query: str) -> str:
        """Analyze with BLIP-2 model"""
        if not query.strip():
            query = "Describe this image in detail"
        
        inputs = self.current_processor(images=image, text=query, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.current_model.generate(**inputs, max_new_tokens=256)
        
        response = self.current_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model"""
        if self.current_model_name is None:
            return {"status": "No model loaded"}
        
        model_info = self.available_models[self.current_model_name].copy()
        model_info.update({
            "status": "Loaded",
            "device": self.device,
            "memory_estimate": self.estimate_memory_usage(self.current_model_name)
        })
        
        return model_info
    
    def unload_model(self):
        """Unload current model to free memory"""
        if self.current_model is not None:
            del self.current_model
            del self.current_processor
            self.current_model = None
            self.current_processor = None
            self.current_model_name = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded and memory cleared")
    
    def is_available(self) -> bool:
        """Check if Hugging Face models are available"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False

# Singleton instance for the app
_hf_processor = None

def get_hf_processor() -> HuggingFaceVisionProcessor:
    """Get singleton HuggingFace processor instance"""
    global _hf_processor
    if _hf_processor is None:
        _hf_processor = HuggingFaceVisionProcessor()
    return _hf_processor
