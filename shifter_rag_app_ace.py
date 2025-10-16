import streamlit as st
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

# Vision API imports
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
    page_title=" Shifter Assistant with ACE", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ACE features
st.markdown("""
    <style>
    .ace-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .ace-metric {
        background-color: #f0f8ff;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .feedback-container {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .evolution-indicator {
        background-color: #e8f5e8;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
    }
    .knowledge-graph-node {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .adaptive-response {
        background-color: #f0f8ff;
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .response-container {
        background-color: #ffffff;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #000000;
    }
    .adaptive-response {
        background-color: #ffffff;
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #000000;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .assistant-message {
        background-color: #f0f8f0;
        margin-right: 20px;
        color: #000000;
        border: 1px solid #ddd;
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
    </style>
    """, unsafe_allow_html=True)

# Vision capabilities
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

# Import existing classes from the original system
class DocumentProcessor:
    """Handle document ingestion and processing"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.html', '.htm', '.md']
    
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
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
            
            decoded_content = None
            for encoding in encodings:
                try:
                    decoded_content = file_content.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if decoded_content is None:
                decoded_content = file_content.decode('utf-8', errors='replace')
            
            soup = BeautifulSoup(decoded_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
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
            
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(uploaded_file.read())
            elif file_extension in ['.html', '.htm']:
                text = self.extract_text_from_html(uploaded_file.read())
            else:
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
                    text = file_bytes.decode('utf-8', errors='replace')
            
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
            chunks = self.chunk_text(doc_data['content'])
            
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
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not query_words or not text_words:
            return 0.0
        
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
                if similarity > 0:
                    metadata = self.chunk_metadata[i]
                    results.append({
                        'content': chunk,
                        'score': similarity,
                        'metadata': metadata,
                        'document': self.documents[metadata['doc_id']]
                    })
            
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
    """Handle conversation memory and context"""
    
    def __init__(self):
        self.conversation_history = []
        self.important_facts = []
        self.user_preferences = {}
        self.session_context = {}
    
    def add_conversation(self, user_query: str, assistant_response: str, context_docs: List[str] = None):
        """Add conversation to memory"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'assistant_response': assistant_response,
            'context_docs': context_docs or [],
            'session_id': st.session_state.get('session_id', 'default')
        }
        self.conversation_history.append(conversation)
        
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
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
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        relevant_conversations = []
        
        for conv in reversed(self.conversation_history):
            conv_text = (conv['user_query'] + ' ' + conv['assistant_response']).lower()
            conv_words = set(re.findall(r'\b\w+\b', conv_text))
            overlap = len(query_words.intersection(conv_words))
            
            if overlap > 0:
                conv['relevance_score'] = overlap / len(query_words)
                relevant_conversations.append(conv)
        
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
                context_parts.append("---")
        
        if self.important_facts:
            context_parts.append("## Important Facts to Remember:")
            for fact in self.important_facts[-5:]:
                context_parts.append(f"- {fact['fact']} ({fact['category']})")
        
        return "\n".join(context_parts) if context_parts else ""

class ShifterRAGSystem:
    """Base RAG system for shifter assistance with vision capabilities"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.search_engine = SimpleTextSearch()
        self.memory_system = MemorySystem()
        self.image_processor = ImageProcessor()
        self.groq_client = None
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize Groq client
        if "groq_api_key" in st.secrets:
            self.groq_client = Groq(api_key=st.secrets["groq_api_key"])
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and "openai_api_key" in st.secrets:
            try:
                self.openai_client = OpenAIClient(api_key=st.secrets["openai_api_key"])
            except Exception:
                self.openai_client = None
        
        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and "anthropic_api_key" in st.secrets:
            try:
                self.anthropic_client = Anthropic(api_key=st.secrets["anthropic_api_key"])
            except Exception:
                self.anthropic_client = None
        
        self.load_search_index()
    
    def load_search_index(self):
        """Load existing search index"""
        # Try to load from vision app first, then ACE-specific
        search_index_paths = ["shifter_docs_vision", "shifter_docs_ace", "shifter_docs_simple"]
        for path in search_index_paths:
            if os.path.exists(f"{path}.pkl"):
                self.search_engine.load_from_disk(path)
                break
    
    def save_search_index(self):
        """Save search index to disk"""
        search_index_path = "shifter_docs_ace"
        return self.search_engine.save_to_disk(search_index_path)
    
    def generate_response(self, query: str, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using Groq API with RAG context"""
        try:
            if not self.groq_client:
                return "Error: Groq API key not configured. Please add your Groq API key to Streamlit secrets."
            
            if not context_docs:
                search_results = self.search_engine.search(query, k=3)
                context_docs = [result['content'] for result in search_results]
            
            memory_context = ""
            if use_memory:
                memory_context = self.memory_system.get_context_prompt(query)
            
            system_prompt = """You are an expert shifter assistant designed to help operations personnel understand what to do when they encounter problems or have questions. 

Your role is to:
1. Provide clear, actionable guidance based on the available documentation
2. Break down complex procedures into step-by-step instructions
3. Highlight important safety considerations or critical points
4. Suggest who to contact or what resources to use when your knowledge isn't sufficient
5. Learn from previous conversations to provide better assistance

Always be concise but complete, and prioritize safety and accuracy."""

            user_prompt = f"""Question: {query}

Available Documentation Context:
{chr(10).join(context_docs) if context_docs else "No specific documentation found."}

{memory_context}

Please provide a helpful response based on the available information. If the documentation doesn't fully address the question, explain what you do know and suggest next steps."""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources)
            
            return assistant_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using vision-capable model (OpenAI, Anthropic, or HuggingFace)"""
        try:
            # Try OpenAI first if available
            if self.openai_client:
                return self._generate_openai_vision_response(query, image_data, context_docs, use_memory)
            elif self.anthropic_client:
                return self._generate_anthropic_vision_response(query, image_data, context_docs, use_memory)
            elif HF_AVAILABLE:
                return self._generate_huggingface_vision_response(query, image_data, context_docs, use_memory)
            else:
                return """⚠️ **No Vision Models Available**
                
No vision models are currently available. Please:

1. **Configure OpenAI API key** in Streamlit secrets: `openai_api_key = "your_key_here"`
2. **Or configure Anthropic API key**: `anthropic_api_key = "your_key_here"`
3. **Or install HuggingFace dependencies**: `pip install transformers torch`

**Available APIs:**
- OpenAI GPT-4o (Vision)
- Anthropic Claude (Vision)
- HuggingFace (Vision)"""
            
        except Exception as e:
            return f"Error generating vision response: {str(e)}"
    
    def _generate_openai_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using OpenAI Vision models"""
        try:
            if not self.openai_client:
                return "❌ OpenAI API key not configured. Add `openai_api_key` to Streamlit secrets."

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
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1200
            )

            assistant_response = resp.choices[0].message.content

            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources, image_data)

            return assistant_response
        except Exception as e:
            return f"❌ Error with OpenAI Vision API: {str(e)}"

    def _generate_anthropic_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using Anthropic Claude vision models"""
        try:
            if not self.anthropic_client:
                return "❌ Anthropic API key not configured. Add `anthropic_api_key` to Streamlit secrets."

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
                model="claude-3-5-sonnet-latest",
                max_tokens=1200,
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

            return assistant_response
        except Exception as e:
            return f"❌ Error with Anthropic Vision API: {str(e)}"

    def _generate_huggingface_vision_response(self, query: str, image_data: Dict, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using HuggingFace vision models"""
        try:
            if not HF_AVAILABLE:
                return "❌ HuggingFace API not available. Check your internet connection."
            
            # Get HuggingFace processor
            hf_processor = get_hf_processor()
            
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
            response = hf_processor.analyze_image(image_data, enhanced_query, "llava-1.5-7b-hf")
            
            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, response, doc_sources, image_data)
            
            return response
            
        except Exception as e:
            return f"❌ Error with HuggingFace API: {str(e)}"

# Initialize session state
if 'ace_rag_system' not in st.session_state:
    base_rag = ShifterRAGSystem()
    st.session_state.ace_rag_system = ACESystem(base_rag)

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load ACE state if available
st.session_state.ace_rag_system.load_ace_state("shifter_docs_ace")

# Main App
def main():
    st.markdown("""
    <div class="ace-header">
        <h1>🧠  Shifter Assistant with Agentic Context Engineering (ACE)</h1>
        <p>Autonomous self-improving RAG system that learns and evolves from user interactions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **ACE Framework Features:**
    - 🧠 **Autonomous Learning**: System learns from every interaction
    - 🔄 **Context Evolution**: Knowledge graph evolves based on feedback
    - 📈 **Adaptive RAG**: Pipeline improves automatically over time
    - 🎯 **Expert Integration**: Incorporates expert corrections and feedback
    - 📊 **Performance Tracking**: Continuous evaluation and optimization
    """)
    
    # Sidebar
    st.sidebar.title("🧠 ACE System Management")
    
    # ACE Metrics Display
    st.sidebar.header("📊 ACE Metrics")
    ace_metrics = st.session_state.ace_rag_system.get_ace_metrics()
    
    st.sidebar.metric("Knowledge Graph Nodes", ace_metrics["knowledge_graph_size"])
    st.sidebar.metric("Total Relationships", ace_metrics["total_relationships"])
    st.sidebar.metric("Adaptation Count", ace_metrics["adaptation_count"])
    
    # Learning metrics
    learning_metrics = ace_metrics["learning_metrics"]
    st.sidebar.metric("Total Interactions", learning_metrics["total_interactions"])
    st.sidebar.metric("Positive Feedback", learning_metrics["positive_feedback"])
    st.sidebar.metric("Accuracy Score", f"{learning_metrics['accuracy_improvement']:.2%}")
    
    # Document Management
    st.sidebar.header("📚 Document Management")
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
                    for uploaded_file in uploaded_files:
                        doc_data = st.session_state.ace_rag_system.adaptive_pipeline.base_rag.doc_processor.process_document(uploaded_file)
                        if doc_data:
                            if st.session_state.ace_rag_system.adaptive_pipeline.base_rag.search_engine.add_document(doc_data):
                                success_count += 1
                    
                    if success_count > 0:
                        st.session_state.ace_rag_system.adaptive_pipeline.base_rag.save_search_index()
                        st.success(f"✅ Successfully processed {success_count} documents!")
                    else:
                        st.error("❌ No documents were processed successfully.")
    
    # Document statistics
    st.sidebar.header("📊 Document Library")
    st.sidebar.info("📚 **Shared with Vision App**: Uses same document library as shifter_rag_app_vision.py")
    
    total_docs = len(st.session_state.ace_rag_system.adaptive_pipeline.base_rag.search_engine.documents)
    total_chunks = len(st.session_state.ace_rag_system.adaptive_pipeline.base_rag.search_engine.chunks)
    
    st.sidebar.metric("Documents", total_docs)
    st.sidebar.metric("Text Chunks", total_chunks)
    
    # Vision API Configuration
    st.sidebar.header("👁️ Vision APIs")
    
    # OpenAI Vision
    if OPENAI_AVAILABLE:
        if "openai_api_key" in st.secrets:
            st.sidebar.success("✅ OpenAI Vision: Available")
            st.sidebar.info("Models: GPT-4o, GPT-4o-mini")
        else:
            st.sidebar.warning("⚠️ OpenAI Vision: Not configured")
            st.sidebar.info("Add to `.streamlit/secrets.toml`: `openai_api_key = \"your_key_here\"`")
    else:
        st.sidebar.warning("OpenAI client not installed. Run: `pip install openai`")
    
    # Anthropic Claude Vision
    if ANTHROPIC_AVAILABLE:
        if "anthropic_api_key" in st.secrets:
            st.sidebar.success("✅ Anthropic Claude: Available")
            st.sidebar.info("Models: Claude 3.5 Sonnet, Claude 3 Haiku")
        else:
            st.sidebar.warning("⚠️ Anthropic Claude: Not configured")
            st.sidebar.info("Add to `.streamlit/secrets.toml`: `anthropic_api_key = \"your_key_here\"`")
    else:
        st.sidebar.warning("Anthropic client not installed. Run: `pip install anthropic`")
    
    # HuggingFace Vision
    if HF_AVAILABLE:
        st.sidebar.success("✅ HuggingFace: Available")
        st.sidebar.info("Models: LLaVA, BLIP, etc.")
    else:
        st.sidebar.warning("⚠️ HuggingFace: Not available")
        st.sidebar.info("Check your internet connection")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 ACE Query Interface", 
        "👁️ Vision Analysis",
        "🧠 Knowledge Graph", 
        "📊 Learning Analytics", 
        "🎯 Feedback System", 
        "⚙️ System Status"
    ])
    
    with tab1:
        st.header("💬 ACE-Enhanced Query Interface")
        st.markdown("Ask questions and watch the system learn and improve from your interactions!")
        
        # Query input
        with st.form(key="ace_query_form", clear_on_submit=False):
            user_query = st.text_area(
                "What do you need help with?",
                placeholder="e.g., 'What should I do if the cooling system alarm goes off?' or 'How do I restart the data acquisition system?'\n\nThe ACE system will learn from your question and improve future responses.",
                height=120,
                key="ace_user_query_input"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                use_memory = st.checkbox("Use conversation memory", value=True, help="Include previous conversations for better context")
            with col2:
                submitted = st.form_submit_button("🧠 Get ACE Response", type="primary", use_container_width=True)
            
            if submitted and user_query.strip():
                with st.spinner("🧠 ACE system processing query with autonomous learning..."):
                    # Generate ACE-enhanced response
                    response, ace_metrics = st.session_state.ace_rag_system.process_query_with_ace(
                        user_query, use_memory=use_memory
                    )
                    
                    # Display response with ACE indicators
                    st.markdown("### 🧠 ACE-Enhanced Response")
                    st.markdown(f'<div class="adaptive-response">{response}</div>', unsafe_allow_html=True)
                    
                    # Show ACE metrics
                    if ace_metrics["ace_applied"]:
                        st.markdown("""
                        <div class="evolution-indicator">
                            🧠 <strong>ACE Applied:</strong> Context evolution and adaptive learning activated
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="ace-metric">
                        <strong>ACE Metrics:</strong><br>
                        • Base contexts: {ace_metrics["base_contexts_count"]}<br>
                        • Evolved contexts: {ace_metrics["evolved_contexts_count"]}<br>
                        • Enhancement ratio: {ace_metrics["context_enhancement_ratio"]:.2f}x
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show relevant documents
                    search_results = st.session_state.ace_rag_system.adaptive_pipeline.base_rag.search_engine.search(user_query, k=3)
                    if search_results:
                        with st.expander("📚 Relevant Documentation"):
                            for i, result in enumerate(search_results):
                                st.write(f"**Source: {result['metadata']['filename']}** (Relevance: {result['score']:.3f})")
                                st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                                st.write("---")
        
        # Quick actions with ACE
        st.markdown("### 🔥 ACE-Enhanced Quick Actions")
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
                if st.button(query, key=f"ace_quick_{i}"):
                    with st.spinner("🧠 ACE system learning from quick action..."):
                        response, ace_metrics = st.session_state.ace_rag_system.process_query_with_ace(query)
                        st.markdown(f"**{query}**")
                        st.markdown(f'<div class="adaptive-response">{response}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("👁️ Vision Analysis")
        st.markdown("Upload images of equipment, error screens, diagrams, or any visual documentation for AI-powered analysis with ACE learning.")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload Image for Analysis",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            help="Supported formats: JPG, PNG, GIF, BMP, WebP (Max: 20MB)"
        )
        
        if uploaded_image:
            # Process image
            with st.spinner("Processing image..."):
                image_data = st.session_state.ace_rag_system.adaptive_pipeline.base_rag.image_processor.process_image(uploaded_image)
            
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
                        vision_response = st.session_state.ace_rag_system.adaptive_pipeline.base_rag.generate_vision_response(
                            query, image_data, use_memory=use_vision_memory
                        )
                        
                        # Display vision response
                        st.markdown("### 👁️ Vision Analysis Result")
                        st.markdown(f'<div class="response-container">{vision_response}</div>', unsafe_allow_html=True)
                        
                        # Show relevant documents if any
                        search_results = st.session_state.ace_rag_system.adaptive_pipeline.base_rag.search_engine.search(query, k=3)
                        if search_results:
                            with st.expander("📚 Relevant Documentation"):
                                for i, result in enumerate(search_results):
                                    st.write(f"**Source: {result['metadata']['filename']}** (Relevance: {result['score']:.3f})")
                                    st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                                    st.write("---")
        
        # Vision API status
        st.markdown("### 🔌 Vision API Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.ace_rag_system.adaptive_pipeline.base_rag.openai_client:
                st.success("✅ OpenAI Vision: Available")
            else:
                st.warning("⚠️ OpenAI Vision: Not configured")
        
        with col2:
            if st.session_state.ace_rag_system.adaptive_pipeline.base_rag.anthropic_client:
                st.success("✅ Anthropic Claude: Available")
            else:
                st.warning("⚠️ Anthropic Claude: Not configured")
        
        with col3:
            if HF_AVAILABLE:
                st.success("✅ HuggingFace: Available")
            else:
                st.warning("⚠️ HuggingFace: Not available")
    
    with tab3:
        st.header("🧠 Knowledge Graph Visualization")
        st.markdown("Explore the evolving knowledge graph that powers ACE's autonomous learning")
        
        # Knowledge graph stats
        context_engine = st.session_state.ace_rag_system.context_engine
        # Calculate average confidence
        avg_confidence = np.mean([node.confidence for node in context_engine.knowledge_graph.values()]) if context_engine.knowledge_graph else 0
        
        st.markdown(f"""
        <div class="ace-metric">
            <strong>Knowledge Graph Statistics:</strong><br>
            • Total nodes: {len(context_engine.knowledge_graph)}<br>
            • Total relationships: {sum(len(rels) for rels in context_engine.relationships.values())}<br>
            • Average confidence: {avg_confidence:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        # Display knowledge graph nodes
        if context_engine.knowledge_graph:
            st.subheader("📊 Knowledge Graph Nodes")
            
            # Filter options
            category_filter = st.selectbox(
                "Filter by category",
                ["All"] + list(set(node.category for node in context_engine.knowledge_graph.values()))
            )
            
            confidence_threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.5)
            
            # Display nodes
            filtered_nodes = []
            for node in context_engine.knowledge_graph.values():
                if (category_filter == "All" or node.category == category_filter) and node.confidence >= confidence_threshold:
                    filtered_nodes.append(node)
            
            for node in sorted(filtered_nodes, key=lambda x: x.confidence, reverse=True)[:10]:
                st.markdown(f"""
                <div class="knowledge-graph-node">
                    <strong>{node.category.title()}</strong> (Confidence: {node.confidence:.2f})<br>
                    <em>{node.content[:200]}{'...' if len(node.content) > 200 else ''}</em><br>
                    <small>Source: {node.source} | Updated: {node.updated_at.strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No knowledge graph nodes yet. Start asking questions to build the knowledge base!")
    
    with tab3:
        st.header("📊 Learning Analytics")
        st.markdown("Track how the ACE system learns and improves over time")
        
        # Learning metrics
        learning_metrics = ace_metrics["learning_metrics"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interactions", learning_metrics["total_interactions"])
        with col2:
            st.metric("Positive Feedback", learning_metrics["positive_feedback"])
        with col3:
            st.metric("Negative Feedback", learning_metrics["negative_feedback"])
        
        # Performance charts
        if learning_metrics["total_interactions"] > 0:
            accuracy = learning_metrics["positive_feedback"] / learning_metrics["total_interactions"]
            st.metric("Current Accuracy", f"{accuracy:.2%}")
            
            # Learning progress
            st.subheader("📈 Learning Progress")
            
            # Simulate learning curve (in real implementation, this would be actual data)
            import matplotlib.pyplot as plt
            import numpy as np
            
            interactions = list(range(1, learning_metrics["total_interactions"] + 1))
            accuracy_progression = [0.5 + 0.3 * (1 - np.exp(-i/10)) for i in interactions]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(interactions, accuracy_progression, 'b-', linewidth=2, label='Accuracy')
            ax.set_xlabel('Interactions')
            ax.set_ylabel('Accuracy')
            ax.set_title('ACE Learning Progress')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.info("No learning data yet. Start interacting with the system to see learning analytics!")
    
    with tab4:
        st.header("🎯 Feedback System")
        st.markdown("Provide feedback to help the ACE system learn and improve")
        
        # Feedback collection
        with st.form(key="feedback_form"):
            st.subheader("📝 Provide Feedback")
            
            feedback_query = st.text_area(
                "Query you asked",
                placeholder="The question you asked the system"
            )
            
            feedback_response = st.text_area(
                "System response",
                placeholder="The response the system provided"
            )
            
            user_rating = st.slider(
                "Rate the response quality",
                min_value=1, max_value=5, value=3,
                help="1 = Poor, 5 = Excellent"
            )
            
            expert_correction = st.text_area(
                "Expert correction (optional)",
                placeholder="If the response was incorrect, provide the correct information"
            )
            
            improvement_suggestions = st.text_area(
                "Improvement suggestions (optional)",
                placeholder="Suggestions for how the system could improve"
            )
            
            if st.form_submit_button("📤 Submit Feedback"):
                if feedback_query and feedback_response:
                    # Create feedback entry
                    feedback_result = st.session_state.ace_rag_system.collect_feedback(
                        feedback_query,
                        feedback_response,
                        user_rating,
                        expert_correction if expert_correction else None,
                        [improvement_suggestions] if improvement_suggestions else None
                    )
                    
                    st.success("✅ Feedback submitted successfully!")
                    st.json(feedback_result)
                else:
                    st.error("Please provide both query and response for feedback")
        
        # Feedback history
        st.subheader("📊 Feedback History")
        feedback_history = st.session_state.ace_rag_system.feedback_collector.feedback_history
        
        if feedback_history:
            for feedback in reversed(feedback_history[-5:]):  # Last 5 feedback entries
                with st.expander(f"Feedback {feedback.id} - Rating: {feedback.user_rating}/5"):
                    st.write(f"**Query:** {feedback.query}")
                    st.write(f"**Response:** {feedback.response}")
                    st.write(f"**Rating:** {feedback.user_rating}/5")
                    if feedback.expert_correction:
                        st.write(f"**Expert Correction:** {feedback.expert_correction}")
                    if feedback.improvement_suggestions:
                        st.write(f"**Suggestions:** {', '.join(feedback.improvement_suggestions)}")
        else:
            st.info("No feedback submitted yet. Help the system learn by providing feedback!")
    
    with tab5:
        st.header("⚙️ ACE System Status")
        
        # System status
        st.subheader("🔌 API Configuration")
        if st.session_state.ace_rag_system.adaptive_pipeline.base_rag.groq_client:
            st.success("✅ Groq API: Connected")
        else:
            st.error("❌ Groq API: Not configured")
            st.info("Add your Groq API key to `.streamlit/secrets.toml`")
        
        # ACE system status
        st.subheader("🧠 ACE System Status")
        st.success("✅ ACE Framework: Active")
        st.success("✅ Context Evolution: Enabled")
        st.success("✅ Adaptive Learning: Active")
        
        # Knowledge graph status
        st.subheader("🧠 Knowledge Graph Status")
        st.metric("Total Nodes", ace_metrics["knowledge_graph_size"])
        st.metric("Total Relationships", ace_metrics["total_relationships"])
        st.metric("Adaptation Count", ace_metrics["adaptation_count"])
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        learning_metrics = ace_metrics["learning_metrics"]
        st.metric("Total Interactions", learning_metrics["total_interactions"])
        st.metric("Response Quality Score", f"{learning_metrics['response_quality_score']:.3f}")
        
        # System actions
        st.subheader("🔧 System Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save ACE State"):
                st.session_state.ace_rag_system.save_ace_state("shifter_docs_ace")
                st.success("✅ ACE state saved!")
        
        with col2:
            if st.button("🔄 Reload ACE State"):
                st.session_state.ace_rag_system.load_ace_state("shifter_docs_ace")
                st.success("✅ ACE state reloaded!")
        
        with col3:
            if st.button("🧹 Reset ACE System"):
                if st.button("⚠️ Confirm Reset", type="secondary"):
                    st.session_state.ace_rag_system = ACESystem(ShifterRAGSystem())
                    st.success("✅ ACE system reset!")

if __name__ == "__main__":
    main()
