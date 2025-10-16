import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from groq import Groq
import os
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pickle
import re
from collections import Counter
import base64
from PIL import Image
import io

# Document processing imports
import PyPDF2
from bs4 import BeautifulSoup
from pathlib import Path

# Note: No longer using local PyTorch models, so no compatibility issues

# Hugging Face Vision Models (API-based)
try:
    from huggingface_vision_api import get_hf_processor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# OpenAI Vision (API-based)
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Anthropic (Claude) Vision (API-based)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import ACE framework
from ace_framework import ACESystem, FeedbackEntry

# Set page config
st.set_page_config(
    page_title=" Shifter Assistant with ACE & Vision", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (enhanced for vision features)
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-text {
        color: #0f9d58;
    }
    .warning-text {
        color: #ff9800;
    }
    .error-text {
        color: #ff4b4b;
    }
    .header-style {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .doc-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f9f9f9;
    }
    .memory-item {
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f8f0;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20px;
    }
    .assistant-message {
        background-color: #f0f8f0;
        margin-right: 20px;
        color: #000000;
        border: 1px solid #ddd;
    }
    .response-container {
        background-color: #ffffff;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #000000;
    }
    .image-container {
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f8f9ff;
    }
    .vision-response {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #000000;
    }
    .ace-metric {
        background-color: #f0f8ff;
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #000000;
    }
    .knowledge-node {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .relationship-item {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 8px;
        margin: 3px 0;
    }
    </style>
    """, unsafe_allow_html=True)

class ImageProcessor:
    """Handle image processing for vision LLM"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        self.max_image_size = 20 * 1024 * 1024  # 20MB max
    
    def validate_image(self, uploaded_file) -> bool:
        """Validate uploaded image"""
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension not in self.supported_formats:
                st.error(f"Unsupported image format: {file_extension}")
                return False
            
            if uploaded_file.size > self.max_image_size:
                st.error(f"Image too large. Max size: {self.max_image_size // (1024*1024)}MB")
                return False
            
            return True
        except Exception as e:
            st.error(f"Error validating image: {str(e)}")
            return False
    
    def process_image(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Process uploaded image for vision LLM"""
        try:
            if not self.validate_image(uploaded_file):
                return None
            
            # Read image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (keep aspect ratio)
            max_dimension = 1024
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create image metadata
            image_data = {
                'filename': uploaded_file.name,
                'format': uploaded_file.type,
                'size': uploaded_file.size,
                'dimensions': image.size,
                'base64': image_base64,
                'upload_time': datetime.now().isoformat()
            }
            
            return image_data
            
        except Exception as e:
            st.error(f"Error processing image {uploaded_file.name}: {str(e)}")
            return None

class DocumentProcessor:
    """Handle document ingestion and processing (enhanced for vision)"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.html', '.htm', '.md']
        self.image_processor = ImageProcessor()
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""
    
    def extract_text_from_html(self, file_content: bytes) -> str:
        """Extract text from HTML file with encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
            
            decoded_content = None
            for encoding in encodings:
                try:
                    decoded_content = file_content.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if decoded_content is None:
                # Last resort: decode with errors='replace'
                decoded_content = file_content.decode('utf-8', errors='replace')
            
            soup = BeautifulSoup(decoded_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            st.error(f"Error processing HTML: {str(e)}")
            return ""
    
    def process_document(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded document and return metadata + text"""
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension not in self.supported_formats:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            # Get file content
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(uploaded_file.read())
            elif file_extension in ['.html', '.htm']:
                text = self.extract_text_from_html(uploaded_file.read())
            else:  # .txt, .md
                # Handle text files with encoding detection
                file_bytes = uploaded_file.read()
                encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
                
                text = None
                for encoding in encodings:
                    try:
                        text = file_bytes.decode(encoding)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                if text is None:
                    # Last resort: decode with errors='replace'
                    text = file_bytes.decode('utf-8', errors='replace')
            
            # Create document metadata
            doc_data = {
                'filename': uploaded_file.name,
                'file_type': file_extension,
                'content': text,
                'upload_time': datetime.now().isoformat(),
                'size': len(text),
                'word_count': len(text.split())
            }
            
            return doc_data
            
        except Exception as e:
            st.error(f"Error processing document {uploaded_file.name}: {str(e)}")
            return None

class SimpleTextSearch:
    """Simple text-based search without machine learning dependencies"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def add_document(self, doc_data: Dict[str, Any]):
        """Add document to search index"""
        try:
            # Chunk the document
            chunks = self.chunk_text(doc_data['content'])
            
            # Store document and chunk metadata
            self.documents.append(doc_data)
            doc_id = len(self.documents) - 1
            
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'doc_id': doc_id,
                    'chunk_id': i,
                    'filename': doc_data['filename'],
                    'file_type': doc_data['file_type']
                })
            
            return True
            
        except Exception as e:
            st.error(f"Error adding document to search index: {str(e)}")
            return False
    
    def calculate_similarity(self, query: str, text: str) -> float:
        """Calculate simple text similarity using word overlap"""
        # Convert to lowercase and split into words
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not query_words or not text_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using simple text matching"""
        try:
            if not self.chunks:
                return []
            
            results = []
            for i, chunk in enumerate(self.chunks):
                similarity = self.calculate_similarity(query, chunk)
                if similarity > 0:  # Only include chunks with some similarity
                    metadata = self.chunk_metadata[i]
                    results.append({
                        'content': chunk,
                        'score': similarity,
                        'metadata': metadata,
                        'document': self.documents[metadata['doc_id']]
                    })
            
            # Sort by similarity score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            st.error(f"Error searching: {str(e)}")
            return []
    
    def save_to_disk(self, path: str):
        """Save search index to disk"""
        try:
            data = {
                'documents': self.documents,
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata
            }
            
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            st.error(f"Error saving search index: {str(e)}")
            return False
    
    def load_from_disk(self, path: str):
        """Load search index from disk"""
        try:
            if os.path.exists(f"{path}.pkl"):
                with open(f"{path}.pkl", 'rb') as f:
                    data = pickle.load(f)
                
                self.documents = data.get('documents', [])
                self.chunks = data.get('chunks', [])
                self.chunk_metadata = data.get('chunk_metadata', [])
            
            return True
        except Exception as e:
            st.error(f"Error loading search index: {str(e)}")
            return False

class MemorySystem:
    """Handle conversation memory and context (enhanced for vision)"""
    
    def __init__(self):
        self.conversation_history = []
        self.important_facts = []
        self.user_preferences = {}
        self.session_context = {}
        self.image_history = []  # Store image analysis history
    
    def add_conversation(self, user_query: str, assistant_response: str, context_docs: List[str] = None, image_data: Dict = None):
        """Add conversation to memory (with optional image)"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'assistant_response': assistant_response,
            'context_docs': context_docs or [],
            'image_data': image_data,  # Store image metadata if provided
            'session_id': st.session_state.get('session_id', 'default')
        }
        self.conversation_history.append(conversation)
        
        # Keep only last 50 conversations to prevent memory bloat
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def add_image_analysis(self, image_data: Dict, analysis: str):
        """Add image analysis to history"""
        image_analysis = {
            'timestamp': datetime.now().isoformat(),
            'filename': image_data['filename'],
            'analysis': analysis,
            'image_metadata': {
                'size': image_data['size'],
                'dimensions': image_data['dimensions'],
                'format': image_data['format']
            }
        }
        self.image_history.append(image_analysis)
        
        # Keep only last 20 image analyses
        if len(self.image_history) > 20:
            self.image_history = self.image_history[-20:]
    
    def add_important_fact(self, fact: str, category: str = "general"):
        """Add important fact to memory"""
        fact_entry = {
            'fact': fact,
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'relevance_count': 1
        }
        self.important_facts.append(fact_entry)
    
    def get_relevant_history(self, query: str, limit: int = 3) -> List[Dict]:
        """Get relevant conversation history based on current query"""
        # Simple keyword matching
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        relevant_conversations = []
        
        for conv in reversed(self.conversation_history):
            conv_text = (conv['user_query'] + ' ' + conv['assistant_response']).lower()
            conv_words = set(re.findall(r'\b\w+\b', conv_text))
            overlap = len(query_words.intersection(conv_words))
            
            if overlap > 0:
                conv['relevance_score'] = overlap / len(query_words)
                relevant_conversations.append(conv)
        
        # Sort by relevance and return top results
        relevant_conversations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_conversations[:limit]
    
    def get_context_prompt(self, query: str) -> str:
        """Generate context prompt from memory"""
        relevant_history = self.get_relevant_history(query)
        
        context_parts = []
        
        if relevant_history:
            context_parts.append("## Recent Relevant Conversations:")
            for conv in relevant_history:
                context_parts.append(f"Q: {conv['user_query']}")
                context_parts.append(f"A: {conv['assistant_response'][:200]}...")
                if conv.get('image_data'):
                    context_parts.append(f"[Previous conversation included image: {conv['image_data']['filename']}]")
                context_parts.append("---")
        
        if self.important_facts:
            context_parts.append("## Important Facts to Remember:")
            for fact in self.important_facts[-5:]:  # Last 5 facts
                context_parts.append(f"- {fact['fact']} ({fact['category']})")
        
        return "\n".join(context_parts) if context_parts else ""

class VisionRAGSystem:
    """Enhanced RAG system with vision capabilities"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.search_engine = SimpleTextSearch()
        self.memory_system = MemorySystem()
        self.image_processor = ImageProcessor()
        self.groq_client = None
        self.openai_client = None
        self.anthropic_client = None
        
        # Vision model providers and their models
        # Note: Groq vision models are currently unavailable
        self.groq_vision_models = {}
        
        # Hugging Face models (if available)
        self.hf_vision_models = {}
        if HF_AVAILABLE:
            hf_processor = get_hf_processor()
            available_models = hf_processor.get_available_models()
            for key, model_info in available_models.items():
                cost_info = hf_processor.estimate_cost(key)
                self.hf_vision_models[f"hf_{key}"] = f"{model_info['name']} (HF API - {cost_info})"

        # OpenAI vision models
        self.openai_vision_models = {}
        if OPENAI_AVAILABLE and "openai_api_key" in st.secrets:
            try:
                self.openai_client = OpenAIClient(api_key=st.secrets["openai_api_key"])
                # Register OpenAI vision-capable models
                self.openai_vision_models = {
                    "oa_gpt-4o-mini": "OpenAI GPT-4o-mini (Vision - Default)",
                    "oa_gpt-4o": "OpenAI GPT-4o (Vision)"
                }
            except Exception:
                self.openai_client = None

        # Anthropic (Claude) vision models
        self.anthropic_vision_models = {}
        if ANTHROPIC_AVAILABLE and "anthropic_api_key" in st.secrets:
            try:
                self.anthropic_client = Anthropic(api_key=st.secrets["anthropic_api_key"])
                # Claude 3.5 Sonnet supports vision
                self.anthropic_vision_models = {
                    "cl_claude-3-5-sonnet-latest": "Claude 3.5 Sonnet (Vision)",
                    "cl_claude-3-haiku-20240307": "Claude 3 Haiku (Vision, budget)"
                }
            except Exception:
                self.anthropic_client = None
        
        # Combined vision models
        self.vision_models = {**self.groq_vision_models, **self.hf_vision_models, **self.openai_vision_models, **getattr(self, 'anthropic_vision_models', {})}
        
        # Vision provider info
        self.vision_providers = {
            "huggingface": "Hugging Face API (Cloud)",
            "openai": "OpenAI API (Vision)",
            "anthropic": "Anthropic Claude (Vision)"
        }
        
        self.text_models = {
            "llama-3.1-8b-instant": "LLaMA 3.1 8B Instant",
            "llama-3.1-70b-versatile": "LLaMA 3.1 70B Versatile",
            "mixtral-8x7b-32768": "Mixtral 8x7B"
        }
        
        # Initialize Groq client
        if "groq_api_key" in st.secrets:
            self.groq_client = Groq(api_key=st.secrets["groq_api_key"])
        
        # Initialize Hugging Face API key
        if "huggingface_api_key" in st.secrets:
            hf_processor = get_hf_processor()
            hf_processor.set_api_key(st.secrets["huggingface_api_key"])

        # Initialize OpenAI (handled above when building models)
        
        # Load existing search index if available
        self.load_search_index()
    
    def load_search_index(self):
        """Load existing search index"""
        search_index_path = "shifter_docs_vision"
        if os.path.exists(f"{search_index_path}.pkl"):
            self.search_engine.load_from_disk(search_index_path)
    
    def save_search_index(self):
        """Save search index to disk"""
        search_index_path = "shifter_docs_vision"
        return self.search_engine.save_to_disk(search_index_path)
    
    def generate_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using vision-capable model (Groq or HuggingFace)"""
        try:
            # Check if any vision models are available
            if not self.vision_models:
                return """⚠️ **No Vision Models Available**
                
No vision models are currently available. Please:

1. **Install Hugging Face dependencies**: `pip install transformers torch`
2. **Or configure Groq API key** in Streamlit secrets
3. **Or try alternative APIs** (OpenAI GPT-4V, Google Gemini)

**Available installation:**
```bash
pip install -r requirements_rag_vision.txt
```"""
            
            # Get model from session state or use first available
            available_models = list(self.vision_models.keys())
            default_model = available_models[0] if available_models else None
            model = st.session_state.get('selected_vision_model', default_model)
            
            # Debug: Show which vision model is being used
            if model:
                st.info(f"👁️ Using vision model: {self.vision_models[model]}")
            
            if not model:
                return "❌ No vision model selected."
            
            # Determine provider based on model prefix
            if model.startswith('hf_'):
                return self._generate_huggingface_response(query, image_data, context_docs, use_memory, model)
            elif model.startswith('oa_'):
                return self._generate_openai_vision_response(query, image_data, context_docs, use_memory, model)
            elif model.startswith('cl_'):
                return self._generate_anthropic_vision_response(query, image_data, context_docs, use_memory, model)
            else:
                # Fallback to Hugging Face if Groq model not available
                if self.hf_vision_models:
                    # Use first available Hugging Face model
                    hf_model = list(self.hf_vision_models.keys())[0]
                    return self._generate_huggingface_response(query, image_data, context_docs, use_memory, hf_model)
                else:
                    return "❌ No vision models available. Please configure Hugging Face API key."
            
        except Exception as e:
            return f"Error generating vision response: {str(e)}"
    
    def _generate_huggingface_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True, model: str = None) -> str:
        """Generate response using Hugging Face API"""
        try:
            if not HF_AVAILABLE:
                return "❌ Hugging Face API not available. Check your internet connection."
            
            # Get HuggingFace processor
            hf_processor = get_hf_processor()
            
            # Extract model key (remove 'hf_' prefix)
            model_key = model[3:] if model.startswith('hf_') else model
            
            # Get relevant documents
            if not context_docs:
                search_results = self.search_engine.search(query, k=3)
                context_docs = [result['content'] for result in search_results]
            
            # Get memory context
            memory_context = ""
            if use_memory:
                memory_context = self.memory_system.get_context_prompt(query)
            
            # Enhance query with context for better API results
            enhanced_query = f"""As a shifter assistant, analyze this image and provide helpful guidance.

Question: {query}

Available Documentation Context:
{chr(10).join(context_docs) if context_docs else "No specific documentation found."}

{memory_context}

Please analyze the image and provide clear, actionable guidance based on both the visual information and available documentation. Focus on safety, procedures, and troubleshooting if relevant."""
            
            # Analyze with HuggingFace API
            response = hf_processor.analyze_image(image_data, enhanced_query, model_key)
            
            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, response, doc_sources, image_data)
                self.memory_system.add_image_analysis(image_data, response)
            
            return response
            
        except Exception as e:
            return f"❌ Error with Hugging Face API: {str(e)}"

    def _generate_openai_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True, model: str = None) -> str:
        """Generate response using OpenAI Vision models (e.g., GPT-4o/-mini)"""
        try:
            if not self.openai_client:
                return "❌ OpenAI API key not configured. Add `openai_api_key` to Streamlit secrets."

            # Extract model key (remove 'oa_' prefix)
            model_key = model[3:] if model and model.startswith('oa_') else (model or "gpt-4o-mini")

            # Get relevant documents
            if not context_docs:
                search_results = self.search_engine.search(query, k=3)
                context_docs = [result['content'] for result in search_results]

            # Get memory context
            memory_context = ""
            if use_memory:
                memory_context = self.memory_system.get_context_prompt(query)

            system_prompt = """You are an expert shifter assistant with vision capabilities. Analyze images and provide:
1) Clear description of what you see
2) Actionable next steps and safety notes
3) Relevant troubleshooting suggestions if applicable"""

            user_text = f"""Question: {query}

Available Documentation Context:
{chr(10).join(context_docs) if context_docs else "No specific documentation found."}

{memory_context}

Please be concise and actionable."""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data['base64']}"}
                        }
                    ]
                }
            ]

            resp = self.openai_client.chat.completions.create(
                model=model_key,
                messages=messages,
                temperature=0.3,
                max_tokens=4000
            )

            assistant_response = resp.choices[0].message.content

            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources, image_data)
                self.memory_system.add_image_analysis(image_data, assistant_response)

            return assistant_response
        except Exception as e:
            # Fallback to Claude if quota/429 and Claude is configured
            if "insufficient_quota" in str(e) or "429" in str(e):
                if self.anthropic_client and getattr(self, 'anthropic_vision_models', {}):
                    cl_model = list(self.anthropic_vision_models.keys())[0]
                    return self._generate_anthropic_vision_response(query, image_data, context_docs, use_memory, cl_model)
            return f"❌ Error with OpenAI Vision API: {str(e)}"

    def _generate_anthropic_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True, model: str = None) -> str:
        """Generate response using Anthropic Claude vision models."""
        try:
            if not self.anthropic_client:
                return "❌ Anthropic API key not configured. Add `anthropic_api_key` to Streamlit secrets."

            model_key = model[3:] if model and model.startswith('cl_') else (model or "claude-3-5-sonnet-latest")

            if not context_docs:
                search_results = self.search_engine.search(query, k=3)
                context_docs = [result['content'] for result in search_results]

            memory_context = ""
            if use_memory:
                memory_context = self.memory_system.get_context_prompt(query)

            user_text = f"""Question: {query}\n\nAvailable Documentation Context:\n{chr(10).join(context_docs) if context_docs else "No specific documentation found."}\n\n{memory_context}\n\nPlease be concise and actionable."""

            content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data['base64']
                    }
                }
            ]

            resp = self.anthropic_client.messages.create(
                model=model_key,
                max_tokens=4000,
                temperature=0.3,
                messages=[{"role": "user", "content": content}]
            )

            # Claude returns a list of content blocks
            assistant_response = "".join(
                block.text for block in getattr(resp, 'content', []) if getattr(block, 'type', '') == 'text'
            ) or str(resp)

            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources, image_data)
                self.memory_system.add_image_analysis(image_data, assistant_response)

            return assistant_response
        except Exception as e:
            return f"❌ Error with Anthropic Vision API: {str(e)}"
    
    def _generate_groq_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True, model: str = None) -> str:
        """Generate response using Groq API model"""
        try:
            if not self.groq_client:
                return "❌ Groq API key not configured. Please add your Groq API key to Streamlit secrets."
            
            # Get relevant documents
            if not context_docs:
                search_results = self.search_engine.search(query, k=3)
                context_docs = [result['content'] for result in search_results]
            
            # Get memory context
            memory_context = ""
            if use_memory:
                memory_context = self.memory_system.get_context_prompt(query)
            
            # Build messages for vision model
            system_prompt = """You are an expert shifter assistant with vision capabilities designed to help operations personnel understand what to do when they encounter problems or have questions. You can analyze images, diagrams, screenshots, and visual documentation.

Your role is to:
1. Analyze images for relevant information (equipment status, error messages, diagrams, etc.)
2. Provide clear, actionable guidance based on both visual and textual documentation
3. Break down complex procedures into step-by-step instructions
4. Highlight important safety considerations or critical points
5. Suggest who to contact or what resources to use when needed
6. Learn from previous conversations to provide better assistance

IMPORTANT: Focus ONLY on the specific question asked. Do not mention unrelated problems or procedures unless directly relevant to the user's question. If the user asks about general procedures, provide general guidance. If they ask about specific problems, provide specific solutions. Always be concise but complete, and prioritize safety and accuracy."""

            # Prepare user message with image
            user_content = [
                {
                    "type": "text",
                    "text": f"""Question: {query}

Available Documentation Context:
{chr(10).join(context_docs) if context_docs else "No specific documentation found."}

{memory_context}

Please analyze the provided image and provide a helpful response based on both the visual information and available documentation. If the image shows equipment, error messages, or diagrams, please describe what you see and provide relevant guidance."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data['base64']}"
                    }
                }
            ]
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources, image_data)
                self.memory_system.add_image_analysis(image_data, assistant_response)
            
            return assistant_response
            
        except Exception as e:
            return f"❌ Error with Groq API: {str(e)}"
    
    def generate_response(self, query: str, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate text-only response using regular LLM"""
        try:
            if not self.groq_client:
                return "Error: Groq API key not configured. Please add your Groq API key to Streamlit secrets."
            
            # Get model from session state or use default
            model = st.session_state.get('selected_text_model', 'llama-3.1-8b-instant')
            
            # Debug: Show which model is being used
            st.info(f"🤖 Using model: {self.text_models[model]}")
            
            # Get relevant documents
            if not context_docs:
                search_results = self.search_engine.search(query, k=3)
                context_docs = [result['content'] for result in search_results]
            
            # Get memory context
            memory_context = ""
            if use_memory:
                memory_context = self.memory_system.get_context_prompt(query)
            
            # Build prompt
            system_prompt = """You are an expert shifter assistant designed to help operations personnel understand what to do when they encounter problems or have questions. 

Your role is to:
1. Provide clear, actionable guidance based on the available documentation
2. Break down complex procedures into step-by-step instructions
3. Highlight important safety considerations or critical points
4. Suggest who to contact or what resources to use when your knowledge isn't sufficient
5. Learn from previous conversations to provide better assistance

IMPORTANT: Focus ONLY on the specific question asked. Do not mention unrelated problems or procedures unless directly relevant to the user's question. If the user asks about general procedures, provide general guidance. If they ask about specific problems, provide specific solutions. Always be concise but complete, and prioritize safety and accuracy."""

            # Build context and compress if needed
            full_context = f"""Available Documentation Context:
{chr(10).join(context_docs) if context_docs else "No specific documentation found."}

{memory_context}"""
            
            # Compress context if using 8B model to avoid token limits
            if model == 'llama-3.1-8b-instant':
                # Use ACE compression if available
                if hasattr(self, 'ace_system') and self.ace_system:
                    full_context = self.ace_system.compress_context_for_api(full_context, max_tokens=4000)
                else:
                    # Simple compression: truncate if too long
                    if len(full_context) > 8000:  # Rough token estimate
                        full_context = full_context[:8000] + "\n\n[Context truncated to fit API limits]"
            
            user_prompt = f"""Question: {query}

{full_context}

Please provide a helpful response based on the available information. Focus ONLY on the specific question asked. Do not mention unrelated problems or procedures unless directly relevant to the user's question. If the user asks about restarting systems, focus on restart procedures. If they ask about general procedures, provide general guidance. If they ask about specific problems, provide specific solutions. IMPORTANT: Do not mention the 4-cells problem unless the user specifically asks about it. If the documentation doesn't fully address the question, explain what you do know and suggest next steps."""

            # Generate response
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources)
            
            return assistant_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize session state with ACE
if 'vision_rag_system' not in st.session_state:
    base_vision_rag = VisionRAGSystem()
    st.session_state.vision_rag_system = ACESystem(base_vision_rag)

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if 'selected_vision_model' not in st.session_state:
    # Prefer OpenAI vision if available; otherwise fall back to Hugging Face
    vr = st.session_state.vision_rag_system.adaptive_pipeline.base_rag
    oa_models = list(getattr(vr, 'openai_vision_models', {}).keys())
    if oa_models:
        st.session_state.selected_vision_model = oa_models[0]
    else:
        hf_models = list(vr.hf_vision_models.keys())
        st.session_state.selected_vision_model = hf_models[0] if hf_models else None

if 'selected_text_model' not in st.session_state:
    st.session_state.selected_text_model = 'llama-3.1-8b-instant'

# Main App
def main():
    st.title("🧠  Shifter Assistant with ACE & Vision 👁️")
    st.markdown("""
    *Autonomous self-improving AI assistant for operational staff - Upload documentation and images, get instant, context-aware help with continuous learning*
    
    🧠 **ACE Framework**: Autonomous learning system that improves from every interaction
    👁️ **Vision Capabilities**: Upload images of equipment, error screens, diagrams, and get AI-powered analysis
    📚 **Shared Document Library**: Uses same documents as your existing vision app
    """)
    
    # Sidebar
    st.sidebar.title("📚 Document & Image Management")
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    vision_model = st.sidebar.selectbox(
        "Vision Model",
        options=list(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.vision_models.keys()),
        format_func=lambda x: st.session_state.vision_rag_system.adaptive_pipeline.base_rag.vision_models[x],
        key="selected_vision_model"
    )
    
    text_model = st.sidebar.selectbox(
        "Text Model",
        options=list(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.text_models.keys()),
        format_func=lambda x: st.session_state.vision_rag_system.adaptive_pipeline.base_rag.text_models[x],
        key="selected_text_model"
    )
    
    # Hugging Face API Configuration
    if HF_AVAILABLE:
        st.sidebar.header("🤗 HuggingFace API")
        
        # Check if API key is configured in secrets
        hf_processor = get_hf_processor()
        api_info = hf_processor.get_model_info()
        
        if api_info.get("api_configured"):
            st.sidebar.success("✅ API Key Configured")
            st.sidebar.info(f"📊 {api_info.get('available_models')} models available")
            st.sidebar.info(f"🆓 {api_info.get('free_models')} free models")
            
            # Test API connection
            if st.sidebar.button("🔍 Test API Connection"):
                with st.spinner("Testing connection..."):
                    test_result = hf_processor.test_api_connection()
                    if "✅" in test_result:
                        st.sidebar.success(test_result)
                    else:
                        st.sidebar.error(test_result)
        else:
            st.sidebar.warning("⚠️ API Key Required")
            st.sidebar.info("Add your Hugging Face API key to `.streamlit/secrets.toml`:")
            st.sidebar.code('huggingface_api_key = "your_api_key_here"')
            
        # Model information
        with st.sidebar.expander("📋 Available Models"):
            available_models = hf_processor.get_available_models()
            
            for model_key, model_info in available_models.items():
                st.write(f"**{model_info['name']}**")
                st.write(f"Type: {model_info['type']}")
                st.write(f"Cost: {hf_processor.estimate_cost(model_key)}")
                st.write(f"_{model_info['description']}_")
                st.write("---")
    else:
        st.sidebar.header("🤗 HuggingFace API")
        st.sidebar.warning("⚠️ HuggingFace API not available")
        st.sidebar.info("Check your internet connection")

    # OpenAI API Configuration
    st.sidebar.header("🧠 OpenAI Vision")
    if OPENAI_AVAILABLE:
        if "openai_api_key" in st.secrets:
            st.sidebar.success("✅ OpenAI API Key Configured")
            st.sidebar.info("Models: GPT-4o-mini, GPT-4o")
        else:
            st.sidebar.warning("⚠️ OpenAI API Key Required")
            st.sidebar.info("Add your OpenAI API key to `.streamlit/secrets.toml`:")
            st.sidebar.code('openai_api_key = "your_api_key_here"')
    else:
        st.sidebar.warning("OpenAI client not installed. Run: pip install openai")

    # Anthropic API Configuration
    st.sidebar.header("🧠 Claude (Anthropic) Vision")
    if ANTHROPIC_AVAILABLE:
        if "anthropic_api_key" in st.secrets:
            st.sidebar.success("✅ Anthropic API Key Configured")
            st.sidebar.info("Models: Claude 3.5 Sonnet, Claude 3 Haiku")
        else:
            st.sidebar.warning("⚠️ Anthropic API Key Required")
            st.sidebar.info("Add your Anthropic API key to `.streamlit/secrets.toml`:")
            st.sidebar.code('anthropic_api_key = "your_api_key_here"')
    else:
        st.sidebar.warning("Anthropic client not installed. Run: pip install anthropic")
    
    # Document upload section
    st.sidebar.header("📄 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documentation files",
        type=['pdf', 'txt', 'html', 'htm', 'md'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, HTML, Markdown"
    )
    
    if uploaded_files:
        with st.sidebar:
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing documents..."):
                    success_count = 0
                    relearning_triggered = False
                    
                    for uploaded_file in uploaded_files:
                        doc_data = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.doc_processor.process_document(uploaded_file)
                        if doc_data:
                            # Add to version control
                            st.session_state.vision_rag_system.context_engine.document_version_control.add_document_version(
                                doc_id=doc_data['filename'],
                                content=doc_data['content'],
                                metadata={
                                    'file_type': doc_data['file_type'],
                                    'upload_time': doc_data['upload_time']
                                }
                            )
                            
                            # Check if relearning is needed
                            if st.session_state.vision_rag_system.context_engine.document_version_control.should_trigger_relearning(doc_data['filename']):
                                relearning_triggered = True
                                st.warning(f"⚠️ Major changes detected in {doc_data['filename']}. System will relearn from this document.")
                            
                            if st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.add_document(doc_data):
                                success_count += 1
                    
                    if success_count > 0:
                        st.session_state.vision_rag_system.adaptive_pipeline.base_rag.save_search_index()
                        st.success(f"✅ Successfully processed {success_count} documents!")
                        
                        if relearning_triggered:
                            st.info("🧠 System is automatically updating knowledge graph based on document changes...")
                    else:
                        st.error("❌ No documents were processed successfully.")
    
    # Document statistics
    st.sidebar.header("📊 Document Library")
    total_docs = len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.documents)
    total_chunks = len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.chunks)
    total_images = len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.image_history)
    
    st.sidebar.metric("Documents", total_docs)
    st.sidebar.metric("Text Chunks", total_chunks)
    st.sidebar.metric("Images Analyzed", total_images)
    
    # ACE Metrics
    st.sidebar.header("🧠 ACE Metrics")
    ace_metrics = st.session_state.vision_rag_system.get_ace_metrics()
    st.sidebar.metric("Knowledge Nodes", ace_metrics["knowledge_graph_size"])
    st.sidebar.metric("Relationships", ace_metrics["total_relationships"])
    st.sidebar.metric("Adaptations", ace_metrics["adaptation_count"])
    
    # Document Refresh Recommendations
    st.sidebar.header("📋 Documentation Needs Update")
    refresh_recs = st.session_state.vision_rag_system.get_document_refresh_recommendations()
    
    if refresh_recs:
        for rec in refresh_recs[:3]:  # Top 3
            with st.sidebar.expander(f"🔴 {rec['topic'].upper()} ({rec['priority']})"):
                st.write(rec['reason'])
                st.write(f"Action: {rec['suggested_action']}")
    else:
        st.sidebar.success("✅ All documentation is up to date!")
    
    if total_docs > 0:
        with st.sidebar.expander("📋 Document List"):
            for i, doc in enumerate(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.documents):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{doc['filename']}**")
                    st.write(f"Type: {doc['file_type']} | Words: {doc['word_count']}")
                    st.write(f"Uploaded: {doc['upload_time'][:10]}")
                with col2:
                    if doc['file_type'] in ['html', 'htm']:
                        if st.button("👁️ View", key=f"view_{i}"):
                            st.session_state.selected_html_doc = doc
                    elif doc['file_type'] == 'pdf':
                        if st.button("📄 Preview", key=f"preview_{i}"):
                            st.session_state.selected_doc_for_analysis = doc
                st.write("---")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["💬 Ask Questions", "👁️ Vision Analysis", "🧠 Knowledge Graph", "🧠 Memory & Context", "📄 Document Viewer", "🔄 Self-Learning", "⚙️ System Status", "🔍 ACE Insights"])
    
    with tab1:
        st.header("Ask the Shifter Assistant")
        
        # Query input with integrated button
        with st.form(key="query_form", clear_on_submit=False):
            user_query = st.text_area(
                "What do you need help with?",
                placeholder="e.g., 'What should I do if the cooling system alarm goes off?' or 'How do I restart the data acquisition system?'\n\nPress Enter or click 'Get Help' to submit your question.",
                height=120,
                key="user_query_input"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                use_memory = st.checkbox("Use conversation memory", value=True, help="Include previous conversations for better context")
            with col2:
                submitted = st.form_submit_button("🚀 Get Help", type="primary", use_container_width=True)
            
            if submitted and user_query.strip():
                with st.spinner("🧠 ACE system learning and generating response..."):
                    # Use ACE-enhanced response generation
                    response, ace_metrics = st.session_state.vision_rag_system.process_query_with_ace(user_query, use_memory=use_memory)
                    
                    # Analyze query patterns for document refresh recommendations
                    response_quality = ace_metrics.get('confidence', 0.5)
                    st.session_state.vision_rag_system.analyze_query_patterns(user_query, response_quality)
                    
                    # Store the response in session state for conversational interface
                    st.session_state.last_text_response = {
                        'query': user_query,
                        'response': response,
                        'ace_metrics': ace_metrics
                    }
                    
                    # Display response with better styling
                    st.markdown("### 🧠 ACE-Enhanced Response")
                    st.markdown(f'<div class="response-container">{response}</div>', unsafe_allow_html=True)
                    
                    # Show ACE metrics if learning occurred
                    if ace_metrics["ace_applied"]:
                        st.markdown("""
                        <div class="evolution-indicator">
                            <strong>🧠 ACE Learning Applied:</strong><br>
                            • Knowledge graph updated<br>
                            • Context refined<br>
                            • System improved for future queries
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show relevant documents
                    search_results = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.search(user_query, k=3)
                    if search_results:
                        with st.expander("📚 Relevant Documentation"):
                            for i, result in enumerate(search_results):
                                st.write(f"**Source: {result['metadata']['filename']}** (Relevance: {result['score']:.3f})")
                                st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                                st.write("---")
        
        # Text conversational interface (outside the form)
        if 'last_text_response' in st.session_state:
            st.markdown("---")
            st.subheader("💬 Continue the Conversation")
            
            response_data = st.session_state.last_text_response
            
            # Feedback collection
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Was this response helpful?**")
                helpful_rating = st.radio(
                    "Rate the response:",
                    ["Very Helpful (5)", "Helpful (4)", "Neutral (3)", "Not Helpful (2)", "Poor (1)"],
                    key="text_rating"
                )
                rating = int(helpful_rating.split("(")[1].split(")")[0])
            
            with col2:
                st.markdown("**Need clarification?**")
                if st.button("🤔 Ask for clarification", key="clarify"):
                    clarification_query = st.text_input(
                        "What would you like me to clarify?",
                        placeholder="e.g., 'Can you explain the restart procedure in more detail?'",
                        key="clarify_input"
                    )
                    if clarification_query:
                        with st.spinner("🧠 ACE system providing clarification..."):
                            clarification_response, _ = st.session_state.vision_rag_system.process_query_with_ace(
                                f"Clarification request: {clarification_query}", use_memory=True
                            )
                            st.markdown("**Clarification:**")
                            st.markdown(f'<div class="response-container">{clarification_response}</div>', unsafe_allow_html=True)
            
            # Expert feedback
            with st.expander("🔧 Expert Feedback (Optional)"):
                expert_correction = st.text_area(
                    "If you're an expert, provide corrections or improvements:",
                    placeholder="e.g., 'The restart procedure should also include checking the cooling system first'",
                    key="expert"
                )
                
                improvement_suggestions = st.text_area(
                    "Suggestions for improvement:",
                    placeholder="e.g., 'Add more specific timing information'",
                    key="improvement"
                )
                
                if st.button("Submit Expert Feedback", key="submit_feedback"):
                    if expert_correction or improvement_suggestions:
                        # Collect feedback
                        feedback_result = st.session_state.vision_rag_system.collect_feedback(
                            query=response_data['query'],
                            response=response_data['response'],
                            user_rating=rating,
                            expert_correction=expert_correction if expert_correction else None,
                            improvement_suggestions=[improvement_suggestions] if improvement_suggestions else None
                        )
                        
                        st.success("✅ Expert feedback collected! The system will learn from your input.")
                        st.info(f"🧠 Learning result: {feedback_result['evolution_result']}")
                        st.info(f"🔧 Adaptation actions: {', '.join(feedback_result['adaptation_actions'])}")
            
            # Follow-up questions
            st.markdown("**💡 Follow-up Questions:**")
            follow_up_questions = [
                "Can you provide more details about the safety procedures?",
                "What should I do if this doesn't work?",
                "Are there any alternative solutions?",
                "What are the warning signs to watch for?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(follow_up_questions):
                with cols[i % 2]:
                    if st.button(f"❓ {question}", key=f"followup_{i}"):
                        with st.spinner("🧠 ACE system answering follow-up..."):
                            follow_up_response, _ = st.session_state.vision_rag_system.process_query_with_ace(
                                question, use_memory=True
                            )
                            st.markdown(f"**Follow-up Answer:**")
                            st.markdown(f'<div class="response-container">{follow_up_response}</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### 🔥 Quick Actions")
        quick_queries = [
            "Emergency shutdown procedures",
            "System restart checklist", 
            "Contact information for on-call personnel",
            "Daily inspection routine",
            "Troubleshooting common alarms"
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(quick_queries):
            with cols[i % 3]:
                if st.button(query, key=f"quick_{i}"):
                    with st.spinner("🧠 ACE system learning..."):
                        response, ace_metrics = st.session_state.vision_rag_system.process_query_with_ace(query)
                        st.markdown(f"**{query}**")
                        st.markdown(f'<div class="chat-message assistant-message">{response}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("👁️ Vision Analysis")
        st.markdown("Upload images of equipment, error screens, diagrams, or any visual documentation for AI-powered analysis.")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload Image for Analysis",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            help="Supported formats: JPG, PNG, GIF, BMP, WebP (Max: 20MB)"
        )
        
        if uploaded_image:
            # Process image
            with st.spinner("Processing image..."):
                image_data = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.image_processor.process_image(uploaded_image)
            
            if image_data:
                st.success("✅ Image processed successfully!")
                # Display image
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(uploaded_image, caption=f"Uploaded: {image_data['filename']}", use_column_width=True)
                
                with col2:
                    st.markdown("**Image Info:**")
                    st.write(f"📁 **File:** {image_data['filename']}")
                    st.write(f"📏 **Size:** {image_data['size']:,} bytes")
                    st.write(f"🖼️ **Dimensions:** {image_data['dimensions'][0]}×{image_data['dimensions'][1]}")
                    st.write(f"🕒 **Uploaded:** {image_data['upload_time'][:16]}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Failed to process image. Please try a different image.")
        
        # Vision analysis form (only show if image was processed successfully)
        if uploaded_image and image_data:
            with st.form(key="vision_form"):
                vision_query = st.text_area(
                    "What would you like to know about this image?",
                    placeholder="e.g., 'What error is shown on this screen?' or 'Describe what you see and suggest next steps' or 'Is this equipment configuration correct?'",
                    height=100
                )
                
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    use_vision_memory = st.checkbox("Use conversation memory", value=True, help="Include previous conversations for better context")
                with col2:
                    vision_submitted = st.form_submit_button("🔍 Analyze Image", type="primary", use_container_width=True)
                with col3:
                    default_submit = st.form_submit_button("📝 Describe what you see and suggest the next steps", use_container_width=True)
                
                if vision_submitted or default_submit:
                    if default_submit and not vision_query.strip():
                        query = "Describe what you see and suggest the next steps."
                    else:
                        query = vision_query.strip() if vision_query.strip() else "Please analyze this image and describe what you see. Provide any relevant guidance or recommendations."
                    
                    with st.spinner("🧠 ACE system analyzing image with vision AI..."):
                        # Use ACE-enhanced vision analysis
                        vision_response, ace_metrics = st.session_state.vision_rag_system.process_vision_query_with_ace(
                            query, image_data, use_memory=use_vision_memory
                        )
                        
                        # Store the analysis in session state for conversational interface
                        st.session_state.last_vision_analysis = {
                            'query': query,
                            'response': vision_response,
                            'ace_metrics': ace_metrics,
                            'image_data': image_data
                        }
                        
                        # Display vision response
                        st.markdown("### 👁️ ACE-Enhanced Vision Analysis")
                        st.markdown(f'<div class="vision-response">{vision_response}</div>', unsafe_allow_html=True)
                        
                        # Show ACE learning indicators
                        if ace_metrics["ace_applied"]:
                            st.markdown("""
                            <div class="evolution-indicator">
                                <strong>🧠 ACE Vision Learning Applied:</strong><br>
                                • Visual knowledge integrated<br>
                                • Context enhanced with image data<br>
                                • System improved for future vision queries
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show relevant documents if any
                        search_results = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.search(query, k=3)
                        if search_results:
                            with st.expander("📚 Relevant Documentation"):
                                for i, result in enumerate(search_results):
                                    st.write(f"**Source: {result['metadata']['filename']}** (Relevance: {result['score']:.3f})")
                                    st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                                    st.write("---")
        
        # Vision conversational interface (outside the form)
        if 'last_vision_analysis' in st.session_state:
            st.markdown("---")
            st.subheader("👁️ Vision Analysis Feedback")
            
            analysis = st.session_state.last_vision_analysis
            
            # Vision feedback collection
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Was the vision analysis accurate?**")
                vision_rating = st.radio(
                    "Rate the analysis:",
                    ["Very Accurate (5)", "Accurate (4)", "Neutral (3)", "Inaccurate (2)", "Very Inaccurate (1)"],
                    key="vision_rating"
                )
                vision_rating_num = int(vision_rating.split("(")[1].split(")")[0])
            
            with col2:
                st.markdown("**Need more analysis?**")
                if st.button("🔍 Analyze further", key="analyze_more"):
                    further_analysis = st.text_input(
                        "What specific aspect would you like me to analyze?",
                        placeholder="e.g., 'Focus on the noisy cells in the bottom left'",
                        key="further_analysis"
                    )
                    if further_analysis:
                        with st.spinner("🧠 ACE system providing detailed analysis..."):
                            detailed_response, _ = st.session_state.vision_rag_system.process_vision_query_with_ace(
                                f"Detailed analysis request: {further_analysis}", analysis['image_data'], use_memory=True
                            )
                            st.markdown("**Detailed Analysis:**")
                            st.markdown(f'<div class="vision-response">{detailed_response}</div>', unsafe_allow_html=True)
            
            # Expert vision feedback
            with st.expander("🔧 Expert Vision Feedback (Optional)"):
                vision_correction = st.text_area(
                    "If you're an expert, provide corrections about the visual analysis:",
                    placeholder="e.g., 'The system should focus on the 4 consecutive vertical cells pattern'",
                    key="vision_expert"
                )
                
                if st.button("Submit Vision Expert Feedback", key="submit_vision_feedback"):
                    if vision_correction:
                        # Collect vision feedback
                        vision_feedback_result = st.session_state.vision_rag_system.collect_feedback(
                            query=analysis['query'],
                            response=analysis['response'],
                            user_rating=vision_rating_num,
                            expert_correction=vision_correction,
                            improvement_suggestions=["Visual analysis improvement"]
                        )
                        
                        st.success("✅ Vision expert feedback collected! The system will learn from your visual analysis expertise.")
                        st.info(f"🧠 Learning result: {vision_feedback_result['evolution_result']}")
                        st.info(f"🔧 Adaptation actions: {', '.join(vision_feedback_result['adaptation_actions'])}")
            
            # Vision-specific follow-up questions
            st.markdown("**🔍 Vision Follow-up Questions:**")
            vision_follow_up = [
                "Can you identify the specific problem areas?",
                "What equipment should I check first?",
                "Are there any safety concerns visible?",
                "What's the recommended action sequence?"
            ]
            
            vision_cols = st.columns(2)
            for i, question in enumerate(vision_follow_up):
                with vision_cols[i % 2]:
                    if st.button(f"👁️ {question}", key=f"vision_followup_{i}"):
                        with st.spinner("🧠 ACE system analyzing follow-up..."):
                            vision_follow_response, _ = st.session_state.vision_rag_system.process_vision_query_with_ace(
                                question, analysis['image_data'], use_memory=True
                            )
                            st.markdown(f"**Vision Follow-up Answer:**")
                            st.markdown(f'<div class="vision-response">{vision_follow_response}</div>', unsafe_allow_html=True)
        
        # Image history
        st.markdown("### 📸 Recent Image Analyses")
        image_history = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.image_history
        
        if image_history:
            for i, analysis in enumerate(reversed(image_history[-5:])):  # Last 5 analyses
                with st.expander(f"🖼️ {analysis['filename']} - {analysis['timestamp'][:16]}"):
                    st.markdown(f"**Analysis:** {analysis['analysis']}")
                    st.markdown(f"**Dimensions:** {analysis['image_metadata']['dimensions'][0]}×{analysis['image_metadata']['dimensions'][1]}")
                    st.markdown(f"**Size:** {analysis['image_metadata']['size']:,} bytes")
        else:
            st.info("No image analyses yet. Upload an image above to get started!")
    
    with tab3:
        st.header("🧠 Knowledge Graph Visualization")
        st.markdown("Explore the evolving knowledge graph that powers ACE's autonomous learning")
        
        # Knowledge graph stats
        context_engine = st.session_state.vision_rag_system.context_engine
        
        # Calculate average confidence
        import numpy as np
        avg_confidence = np.mean([node.confidence for node in context_engine.knowledge_graph.values()]) if context_engine.knowledge_graph else 0
        
        st.markdown(f"""
        <div class="ace-metric">
            <strong>Knowledge Graph Statistics:</strong><br>
            • Total nodes: {len(context_engine.knowledge_graph)}<br>
            • Total relationships: {sum(len(rels) for rels in context_engine.relationships.values())}<br>
            • Average confidence: {avg_confidence:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        # Knowledge graph nodes display
        if context_engine.knowledge_graph:
            st.subheader("📊 Knowledge Nodes")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                node_types = list(set([node.category for node in context_engine.knowledge_graph.values()]))
                selected_type = st.selectbox("Filter by Type", ["All"] + node_types)
            with col2:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
            with col3:
                sort_by = st.selectbox("Sort by", ["Confidence", "Created", "Updated", "Usage"])
            
            # Filter and sort nodes
            filtered_nodes = []
            for node_id, node in context_engine.knowledge_graph.items():
                if selected_type == "All" or node.category == selected_type:
                    if node.confidence >= min_confidence:
                        filtered_nodes.append((node_id, node))
            
            # Sort nodes
            if sort_by == "Confidence":
                filtered_nodes.sort(key=lambda x: x[1].confidence, reverse=True)
            elif sort_by == "Created":
                filtered_nodes.sort(key=lambda x: x[1].created_at, reverse=True)
            elif sort_by == "Updated":
                filtered_nodes.sort(key=lambda x: x[1].updated_at, reverse=True)
            elif sort_by == "Usage":
                filtered_nodes.sort(key=lambda x: x[1].usage_count, reverse=True)
            
            # Display nodes
            for node_id, node in filtered_nodes[:20]:  # Show top 20
                with st.expander(f"🔗 {node.category.title()} - Confidence: {node.confidence:.2f}"):
                    st.markdown(f"**Content:** {node.content[:200]}{'...' if len(node.content) > 200 else ''}")
                    st.markdown(f"**Source:** {node.source}")
                    st.markdown(f"**Created:** {node.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**Updated:** {node.updated_at.strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**Usage Count:** {node.usage_count}")
                    st.markdown(f"**Feedback Score:** {node.feedback_score:.2f}")
                    
                    # Show relationships
                    if node_id in context_engine.relationships and context_engine.relationships[node_id]:
                        st.markdown("**Connected to:**")
                        for related_id in context_engine.relationships[node_id][:5]:  # Show first 5
                            if related_id in context_engine.knowledge_graph:
                                related_node = context_engine.knowledge_graph[related_id]
                                st.markdown(f"• {related_node.category}: {related_node.content[:50]}...")
        else:
            st.info("No knowledge nodes yet. Start asking questions to build the knowledge graph!")
        
        # Visual knowledge graph (simplified)
        st.subheader("🔗 Knowledge Graph Relationships")
        if context_engine.relationships:
            # Create a simple relationship visualization
            relationship_data = []
            for node_id, related_nodes in context_engine.relationships.items():
                if related_nodes:
                    for related_id in related_nodes:
                        if related_id in context_engine.knowledge_graph:
                            source_node = context_engine.knowledge_graph[node_id]
                            target_node = context_engine.knowledge_graph[related_id]
                            relationship_data.append({
                                'source': f"{source_node.category}_{node_id[:8]}",
                                'target': f"{target_node.category}_{related_id[:8]}",
                                'source_type': source_node.category,
                                'target_type': target_node.category
                            })
            
            if relationship_data:
                st.markdown("**Knowledge Graph Relationships:**")
                for rel in relationship_data[:10]:  # Show first 10 relationships
                    st.markdown(f"• {rel['source_type']} → {rel['target_type']}")
            else:
                st.info("No relationships found yet. The system will create connections as it learns.")
        else:
            st.info("No relationships yet. The system will create connections as it learns.")
        
        # Learning metrics
        st.subheader("📈 Learning Metrics")
        learning_metrics = context_engine.learning_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Knowledge Growth", f"{learning_metrics.knowledge_growth_rate:.2f}")
        with col2:
            st.metric("Accuracy Improvement", f"{learning_metrics.accuracy_improvement:.2f}")
        with col3:
            st.metric("Response Quality", f"{learning_metrics.response_quality_score:.2f}")
        with col4:
            st.metric("Total Interactions", learning_metrics.total_interactions)
    
    with tab4:
        st.header("🧠 Memory & Context Management")
        
        # Conversation history
        st.subheader("Recent Conversations")
        conversations = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.conversation_history[-10:]  # Last 10
        
        if conversations:
            for conv in reversed(conversations):
                with st.expander(f"Q: {conv['user_query'][:60]}... ({conv['timestamp'][:16]})"):
                    st.markdown(f"**Question:** {conv['user_query']}")
                    st.markdown(f"**Response:** {conv['assistant_response']}")
                    if conv.get('image_data'):
                        st.markdown(f"**🖼️ Image:** {conv['image_data']['filename']} ({conv['image_data']['dimensions'][0]}×{conv['image_data']['dimensions'][1]})")
                    if conv['context_docs']:
                        st.markdown(f"**Sources:** {', '.join(conv['context_docs'])}")
        else:
            st.info("No conversation history yet. Start asking questions!")
        
        # Important facts management
        st.subheader("Important Facts")
        
        # Add new fact
        with st.expander("➕ Add Important Fact"):
            new_fact = st.text_area("Fact description", placeholder="e.g., 'Always check pressure readings before starting pump A'")
            fact_category = st.selectbox("Category", ["general", "safety", "procedures", "contacts", "troubleshooting"])
            
            if st.button("Add Fact"):
                if new_fact.strip():
                    st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.add_important_fact(new_fact, fact_category)
                    st.success("✅ Fact added to memory!")
        
        # Add 4-cells problem knowledge
        with st.expander("🔧 Add 4-Cells Problem Knowledge"):
            st.markdown("**Add specific operational knowledge about the 4-cells problem:**")
            four_cells_fact = st.text_area(
                "4-Cells Problem Response", 
                value="When 4 consecutive vertical cells turn noisy like the examples below, restart the affected system (ECAL or HCAL), or only the specific board if you can find it without losing too much time of data taking with this problem. Report on the ProblemDQ with the affected run/s.",
                height=100
            )
            
            if st.button("Add 4-Cells Knowledge"):
                if four_cells_fact.strip():
                    # Add to memory system
                    st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.add_important_fact(
                        four_cells_fact, "troubleshooting"
                    )
                    
                    # Also add to ACE knowledge graph for better learning
                    ace_system = st.session_state.vision_rag_system
                    ace_system.context_engine.add_context_node(
                        content=four_cells_fact,
                        category="4cells_problem",
                        source="expert_knowledge",
                        confidence=1.0
                    )
                    
                    st.success("✅ 4-Cells problem knowledge added to ACE system!")
                    st.info("🧠 This knowledge will now be used for future 4-cells problem analysis!")
        
        # Display important facts
        facts = st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.important_facts
        if facts:
            for fact in reversed(facts[-10:]):  # Last 10 facts
                st.markdown(f'<div class="memory-item"><strong>{fact["category"].title()}:</strong> {fact["fact"]}<br><small>{fact["timestamp"][:16]}</small></div>', unsafe_allow_html=True)
        else:
            st.info("No important facts stored yet.")
        
        # Memory cleanup
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Conversation History"):
                st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.conversation_history = []
                st.success("Conversation history cleared!")
        
        with col2:
            if st.button("🖼️ Clear Image History"):
                st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.image_history = []
                st.success("Image history cleared!")
    
    with tab6:
        st.header("🔄 Self-Learning & Document Management")
        
        # Document Refresh Recommendations
        st.subheader("📋 Document Refresh Recommendations")
        refresh_recs = st.session_state.vision_rag_system.get_document_refresh_recommendations()
        
        if refresh_recs:
            for i, rec in enumerate(refresh_recs):
                with st.expander(f"🔴 {rec['topic'].upper()} - {rec['priority'].upper()} Priority"):
                    st.write(f"**Reason:** {rec['reason']}")
                    st.write(f"**Suggested Action:** {rec['suggested_action']}")
                    
                    if st.button(f"Mark as Addressed", key=f"address_{i}"):
                        st.success(f"✅ Marked {rec['topic']} as addressed")
        else:
            st.success("✅ No document refresh recommendations at this time!")
        
        # Contradiction Detection
        st.subheader("⚠️ Knowledge Contradictions")
        
        # Check for contradictions in recent knowledge
        contradiction_count = len(st.session_state.vision_rag_system.context_engine.contradiction_log)
        
        if contradiction_count > 0:
            st.warning(f"Found {contradiction_count} potential contradictions in knowledge base")
            
            with st.expander("View Contradictions"):
                for i, contradiction in enumerate(st.session_state.vision_rag_system.context_engine.contradiction_log[-5:]):
                    st.write(f"**Contradiction {i+1}:**")
                    st.write(f"Timestamp: {contradiction['timestamp']}")
                    st.write(f"Reason: {contradiction.get('reason', 'Unknown')}")
                    if 'existing_node' in contradiction:
                        st.write(f"Conflicts with: {contradiction['existing_node']}")
        else:
            st.success("✅ No contradictions detected in knowledge base!")
        
        # Confidence Decay Management
        st.subheader("📉 Knowledge Confidence Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Confidence Decay"):
                with st.spinner("Applying confidence decay to old knowledge..."):
                    st.session_state.vision_rag_system.apply_confidence_decay()
                    st.success("✅ Confidence decay applied to old knowledge!")
        
        with col2:
            if st.button("Review Low Confidence Knowledge"):
                low_confidence_nodes = [
                    node for node in st.session_state.vision_rag_system.context_engine.knowledge_graph.values()
                    if node.confidence < 0.3
                ]
                
                if low_confidence_nodes:
                    st.warning(f"Found {len(low_confidence_nodes)} low-confidence knowledge nodes")
                    for node in low_confidence_nodes[:3]:
                        st.write(f"• {node.content[:100]}... (Confidence: {node.confidence:.2f})")
                else:
                    st.success("✅ All knowledge nodes have good confidence levels!")
        
        # Multimodal Learning Status
        st.subheader("👁️ Multimodal Learning Status")
        
        visual_patterns = len(st.session_state.vision_rag_system.multimodal_learning_engine.visual_patterns)
        text_visual_mappings = len(st.session_state.vision_rag_system.multimodal_learning_engine.text_visual_mappings)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Visual Patterns Learned", visual_patterns)
        with col2:
            st.metric("Text-Visual Mappings", text_visual_mappings)
        
        if visual_patterns > 0:
            st.info("🧠 System has learned visual patterns from expert feedback!")
        else:
            st.info("💡 Upload images with expert corrections to enable visual learning!")
        
        # System Learning Actions
        st.subheader("🔧 Learning Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Trigger Knowledge Graph Update"):
                with st.spinner("Updating knowledge graph..."):
                    # Apply any pending updates
                    st.session_state.vision_rag_system.apply_confidence_decay()
                    st.success("✅ Knowledge graph updated!")
        
        with col2:
            if st.button("📊 Generate Learning Report"):
                with st.spinner("Generating learning report..."):
                    ace_metrics = st.session_state.vision_rag_system.get_ace_metrics()
                    
                    st.markdown("### 📊 Learning Report")
                    st.write(f"**Knowledge Nodes:** {ace_metrics['knowledge_graph_size']}")
                    st.write(f"**Relationships:** {ace_metrics['total_relationships']}")
                    st.write(f"**Adaptations:** {ace_metrics['adaptation_count']}")
                    st.write(f"**Contradictions:** {len(st.session_state.vision_rag_system.context_engine.contradiction_log)}")
                    st.write(f"**Visual Patterns:** {visual_patterns}")
    
    with tab7:
        st.header("⚙️ System Status")
        
        # API status
        st.subheader("🔌 API Configuration")
        
        # Groq API status (for text models)
        if st.session_state.vision_rag_system.adaptive_pipeline.base_rag.groq_client:
            st.success("✅ Groq API: Connected (Text Models)")
            st.info(f"🤖 **Text Model:** {st.session_state.vision_rag_system.adaptive_pipeline.base_rag.text_models[st.session_state.selected_text_model]}")
        else:
            st.warning("⚠️ Groq API: Not configured (Text models unavailable)")
            st.info("Add your Groq API key to `.streamlit/secrets.toml`:")
            st.code('groq_api_key = "your_api_key_here"')
        
        # Hugging Face API status (for vision models)
        hf_processor = get_hf_processor()
        hf_info = hf_processor.get_model_info()
        if hf_info.get("api_configured"):
            st.success("✅ Hugging Face API: Connected (Vision Models)")
            if st.session_state.selected_vision_model:
                st.info(f"👁️ **Vision Model:** {st.session_state.vision_rag_system.adaptive_pipeline.base_rag.vision_models[st.session_state.selected_vision_model]}")
        else:
            st.warning("⚠️ Hugging Face API: Not configured (Vision models unavailable)")
            st.info("Add your Hugging Face API key to `.streamlit/secrets.toml`")
        
        # Search engine status
        st.subheader("🔍 Search Engine Status")
        st.info("ℹ️ Using simplified text search (no ML dependencies)")
        st.metric("Total Documents", len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.documents))
        st.metric("Text Chunks", len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.search_engine.chunks))
        
        # Vision capabilities
        st.subheader("👁️ Vision Capabilities")
        st.success("✅ Vision processing enabled")
        st.metric("Images Analyzed", len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.image_history))
        
        # Memory status
        st.subheader("🧠 Memory Status")
        st.metric("Conversations", len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.conversation_history))
        st.metric("Important Facts", len(st.session_state.vision_rag_system.adaptive_pipeline.base_rag.memory_system.important_facts))
        
        # System actions
        st.subheader("🔧 System Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Search Index"):
                if st.session_state.vision_rag_system.save_search_index():
                    st.success("✅ Search index saved!")
                else:
                    st.error("❌ Failed to save search index")
        
        with col2:
            if st.button("🔄 Reload Search Index"):
                st.session_state.vision_rag_system.load_search_index()
                st.success("✅ Search index reloaded!")
    
    with tab6:
        st.header("📄 Document Viewer & Improvement Suggestions")
        
        # HTML Document Viewer
        if 'selected_html_doc' in st.session_state and st.session_state.selected_html_doc:
            doc = st.session_state.selected_html_doc
            st.subheader(f"👁️ Viewing: {doc['filename']}")
            
            # Display HTML content
            st.markdown("**HTML Content:**")
            components.html(doc['content'], height=600, scrolling=True)
            
            # Document analysis and suggestions
            st.markdown("---")
            st.subheader("🔍 Document Analysis & Improvement Suggestions")
            
            # Analyze document content
            content = doc['content']
            word_count = len(content.split())
            char_count = len(content)
            
            # Basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", word_count)
            with col2:
                st.metric("Character Count", char_count)
            with col3:
                st.metric("File Type", doc['file_type'].upper())
            
            # Generate improvement suggestions using ACE
            if st.button("🧠 Generate Improvement Suggestions"):
                with st.spinner("Analyzing document with ACE system..."):
                    # Create analysis prompt
                    analysis_prompt = f"""
                    Analyze this HTML document for  shifter operations and provide improvement suggestions:
                    
                    Document: {doc['filename']}
                    Content: {content[:2000]}...
                    
                    Please provide:
                    1. Content quality assessment
                    2. Missing information suggestions
                    3. Structure improvements
                    4. Clarity enhancements
                    5. Operational relevance
                    """
                    
                    # Use ACE system to analyze
                    analysis_response, ace_metrics = st.session_state.vision_rag_system.process_query_with_ace(
                        analysis_prompt, use_memory=True
                    )
                    
                    st.markdown("**📊 Document Analysis:**")
                    st.markdown(f'<div class="response-container">{analysis_response}</div>', unsafe_allow_html=True)
                    
                    # Show ACE learning indicators
                    if ace_metrics.get('ace_applied', False):
                        st.info("🧠 ACE system applied learning to document analysis")
                        if ace_metrics.get('expert_knowledge_used', False):
                            st.success("✅ Expert knowledge used in analysis")
        
        # PDF Document Preview
        elif 'selected_doc_for_analysis' in st.session_state and st.session_state.selected_doc_for_analysis:
            doc = st.session_state.selected_doc_for_analysis
            st.subheader(f"📄 Analyzing: {doc['filename']}")
            
            # Show document metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", doc['word_count'])
            with col2:
                st.metric("File Size", f"{doc['size']:,} bytes")
            with col3:
                st.metric("Upload Date", doc['upload_time'][:10])
            
            # Show document content preview
            st.markdown("**Content Preview:**")
            content_preview = doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content']
            st.text_area("Document Content", content_preview, height=300, disabled=True)
            
            # Generate improvement suggestions
            if st.button("🧠 Analyze PDF Document"):
                with st.spinner("Analyzing PDF document with ACE system..."):
                    analysis_prompt = f"""
                    Analyze this PDF document for  shifter operations:
                    
                    Document: {doc['filename']}
                    Content: {doc['content'][:2000]}...
                    
                    Provide improvement suggestions for:
                    1. Content organization
                    2. Missing procedures
                    3. Clarity improvements
                    4. Operational relevance
                    """
                    
                    analysis_response, ace_metrics = st.session_state.vision_rag_system.process_query_with_ace(
                        analysis_prompt, use_memory=True
                    )
                    
                    st.markdown("**📊 Document Analysis:**")
                    st.markdown(f'<div class="response-container">{analysis_response}</div>', unsafe_allow_html=True)
        
        # Document improvement suggestions
        st.markdown("---")
        st.subheader("💡 General Document Improvement Tips")
        
        improvement_tips = [
            "**Structure**: Use clear headings and sections for easy navigation",
            "**Procedures**: Include step-by-step instructions with safety warnings",
            "**Diagrams**: Add visual aids for complex procedures",
            "**Troubleshooting**: Include common problems and solutions",
            "**Contacts**: List relevant personnel and emergency contacts",
            "**Updates**: Include version numbers and last updated dates",
            "**Accessibility**: Use clear language and avoid jargon",
            "**Searchability**: Include relevant keywords and tags"
        ]
        
        for tip in improvement_tips:
            st.markdown(f"• {tip}")
        
        # Clear selection buttons
        if st.button("🔄 Clear Document Selection"):
            if 'selected_html_doc' in st.session_state:
                del st.session_state.selected_html_doc
            if 'selected_doc_for_analysis' in st.session_state:
                del st.session_state.selected_doc_for_analysis
            st.success("Document selection cleared!")
    
    with tab8:
        st.header("🔍 ACE Insights - Generator-Reflector-Curator Pipeline")
        st.markdown("**See how the ACE framework reasons, reflects, and curates knowledge in real-time**")
        
        # ACE Pipeline Demo
        st.subheader("🧪 ACE Pipeline Demonstration")
        
        demo_query = st.text_input(
            "Enter a query to see ACE pipeline in action:",
            value="What is the 4-cells problem?",
            help="This will show you how the Generator creates deltas, Reflector analyzes them, and Curator organizes knowledge"
        )
        
        if st.button("🔍 Run ACE Pipeline Demo", type="primary"):
            with st.spinner("Running ACE Generator-Reflector-Curator pipeline..."):
                # Run ACE pipeline demonstration
                demo_result = st.session_state.vision_rag_system.demonstrate_ace_pipeline(demo_query)
                
                st.success("✅ ACE Pipeline Demo Complete!")
                
                # Display Generator Output
                st.subheader("🔧 Generator Output (Grow Principle)")
                generator_output = demo_result['generator_output']
                st.write(f"**Deltas Generated:** {generator_output['deltas_count']}")
                
                if generator_output['deltas']:
                    st.write("**Generated Deltas:**")
                    for i, delta in enumerate(generator_output['deltas'], 1):
                        with st.expander(f"Delta {i}: {delta['type'].title()} - {delta['content'][:50]}..."):
                            st.write(f"**Type:** {delta['type']}")
                            st.write(f"**Content:** {delta['content']}")
                            st.write(f"**ID:** {delta['id']}")
                
                # Display Reflector Output
                st.subheader("🔍 Reflector Output (Analysis)")
                reflector_output = demo_result['reflector_output']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Context Quality Score", f"{reflector_output['context_quality_score']:.2f}")
                    st.metric("Execution Effectiveness", f"{reflector_output['execution_effectiveness']:.2f}")
                
                with col2:
                    st.metric("Insights Generated", len(reflector_output['insights']))
                    st.metric("Recommendations", len(reflector_output['recommendations']))
                
                if reflector_output['insights']:
                    st.write("**Insights:**")
                    for insight in reflector_output['insights']:
                        st.write(f"• {insight}")
                
                if reflector_output['recommendations']:
                    st.write("**Recommendations:**")
                    for rec in reflector_output['recommendations']:
                        st.write(f"• {rec}")
                
                # Display Curator Output
                st.subheader("📚 Curator Output (Refine Principle)")
                curator_output = demo_result['curator_output']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Context Updates", curator_output['context_updates'])
                with col2:
                    st.metric("Anti-Collapse Actions", len(curator_output['anti_collapse_actions']))
                with col3:
                    st.metric("Organization Actions", len(curator_output['organization_actions']))
                
                if curator_output['anti_collapse_actions']:
                    st.write("**Anti-Collapse Actions:**")
                    for action in curator_output['anti_collapse_actions']:
                        st.write(f"• {action}")
                
                if curator_output['organization_actions']:
                    st.write("**Organization Actions:**")
                    for action in curator_output['organization_actions']:
                        st.write(f"• {action}")
        
        # Real-time ACE Metrics
        st.subheader("📊 Real-time ACE Metrics")
        
        ace_metrics = st.session_state.vision_rag_system.get_ace_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Generation Cycles", ace_metrics.get('generation_cycles', 0))
            st.metric("Deltas Generated", ace_metrics.get('deltas_generated_total', 0))
        with col2:
            st.metric("Reflection Cycles", ace_metrics.get('reflection_cycles', 0))
            st.metric("Insights Extracted", ace_metrics.get('insights_extracted_total', 0))
        with col3:
            st.metric("Curation Cycles", ace_metrics.get('curation_cycles', 0))
            st.metric("Context Updates", ace_metrics.get('context_updates_total', 0))
        with col4:
            st.metric("Anti-Collapse Actions", ace_metrics.get('anti_collapse_actions_total', 0))
            st.metric("Knowledge Growth Rate", f"{ace_metrics.get('learning_metrics', {}).get('knowledge_growth_rate', 0):.2f}")
        
        # ACE Pipeline Status
        st.subheader("🎯 ACE Pipeline Status")
        status = st.session_state.vision_rag_system.get_ace_status()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔧 Generator Status**")
            st.success(f"Status: {status.get('pipeline_components', {}).get('generator', {}).get('status', 'unknown')}")
            st.info(f"Cycles: {status.get('pipeline_components', {}).get('generator', {}).get('generation_cycles', 0)}")
            st.info(f"History: {status.get('pipeline_components', {}).get('generator', {}).get('history_size', 0)} entries")
        
        with col2:
            st.markdown("**🔍 Reflector Status**")
            st.success(f"Status: {status.get('pipeline_components', {}).get('reflector', {}).get('status', 'unknown')}")
            st.info(f"Cycles: {status.get('pipeline_components', {}).get('reflector', {}).get('reflection_cycles', 0)}")
            st.info(f"History: {status.get('pipeline_components', {}).get('reflector', {}).get('history_size', 0)} entries")
        
        with col3:
            st.markdown("**📚 Curator Status**")
            st.success(f"Status: {status.get('pipeline_components', {}).get('curator', {}).get('status', 'unknown')}")
            st.info(f"Cycles: {status.get('pipeline_components', {}).get('curator', {}).get('curation_cycles', 0)}")
            st.info(f"History: {status.get('pipeline_components', {}).get('curator', {}).get('history_size', 0)} entries")
        
        # Recent ACE Activity
        st.subheader("📈 Recent ACE Activity")
        
        if hasattr(st.session_state.vision_rag_system, 'execution_traces') and st.session_state.vision_rag_system.execution_traces:
            st.write("**Recent Execution Traces:**")
            for i, trace in enumerate(reversed(st.session_state.vision_rag_system.execution_traces[-5:]), 1):
                with st.expander(f"Trace {i}: {trace.query[:50]}..."):
                    st.write(f"**Query:** {trace.query}")
                    st.write(f"**Execution Time:** {trace.execution_time:.2f}s")
                    st.write(f"**Success Indicators:** {', '.join(trace.success_indicators) if trace.success_indicators else 'None'}")
                    st.write(f"**Failure Indicators:** {', '.join(trace.failure_indicators) if trace.failure_indicators else 'None'}")
                    st.write(f"**Timestamp:** {trace.timestamp}")
        else:
            st.info("No execution traces yet. Ask some questions to see ACE in action!")
        
        # Expert Knowledge Retrieval
        st.subheader("🧠 Expert Knowledge Retrieval")
        
        test_query = st.text_input(
            "Test expert knowledge retrieval:",
            value="4-cells problem",
            help="See what expert knowledge the system has learned"
        )
        
        if st.button("🔍 Retrieve Expert Knowledge"):
            expert_knowledge = st.session_state.vision_rag_system._retrieve_expert_knowledge(test_query)
            
            if expert_knowledge:
                st.success("✅ Expert knowledge found!")
                st.markdown(f"**Retrieved Knowledge:** {expert_knowledge}")
            else:
                st.info("No expert knowledge found for this query. Provide some expert feedback to build the knowledge base!")
        
        # ACE Learning Indicators
        st.subheader("🎯 ACE Learning Indicators")
        
        if 'last_text_response' in st.session_state:
            response_data = st.session_state.last_text_response
            ace_metrics = response_data.get('ace_metrics', {})
            
            if ace_metrics.get('ace_pipeline_executed'):
                st.success("✅ ACE Pipeline was executed in the last query!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generation Cycles", ace_metrics.get('generation_cycles', 0))
                with col2:
                    st.metric("Reflection Cycles", ace_metrics.get('reflection_cycles', 0))
                with col3:
                    st.metric("Curation Cycles", ace_metrics.get('curation_cycles', 0))
                
                if ace_metrics.get('expert_knowledge_used'):
                    st.success("🧠 Expert knowledge was used in the response!")
                else:
                    st.info("💡 No expert knowledge was used. Provide expert feedback to enhance responses!")
            else:
                st.warning("⚠️ ACE Pipeline was not executed in the last query.")
        else:
            st.info("No recent queries. Ask a question to see ACE learning indicators!")

if __name__ == "__main__":
    main()
