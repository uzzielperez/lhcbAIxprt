import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import os
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any
import pickle
import re
from collections import Counter

# Document processing imports
import PyPDF2
from bs4 import BeautifulSoup
from pathlib import Path
import io

# Set page config
st.set_page_config(
    page_title="Shifter Assistant RAG System", 
    page_icon="🔧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    </style>
    """, unsafe_allow_html=True)

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
        
        # Keep only last 50 conversations to prevent memory bloat
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
                context_parts.append("---")
        
        if self.important_facts:
            context_parts.append("## Important Facts to Remember:")
            for fact in self.important_facts[-5:]:  # Last 5 facts
                context_parts.append(f"- {fact['fact']} ({fact['category']})")
        
        return "\n".join(context_parts) if context_parts else ""

class ShifterRAGSystem:
    """Main RAG system for shifter assistance"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.search_engine = SimpleTextSearch()
        self.memory_system = MemorySystem()
        self.groq_client = None
        
        # Initialize Groq client
        if "groq_api_key" in st.secrets:
            self.groq_client = Groq(api_key=st.secrets["groq_api_key"])
        
        # Load existing search index if available
        self.load_search_index()
    
    def load_search_index(self):
        """Load existing search index"""
        search_index_path = "shifter_docs_simple"
        if os.path.exists(f"{search_index_path}.pkl"):
            self.search_engine.load_from_disk(search_index_path)
    
    def save_search_index(self):
        """Save search index to disk"""
        search_index_path = "shifter_docs_simple"
        return self.search_engine.save_to_disk(search_index_path)
    
    def generate_response(self, query: str, context_docs: List[str] = None, use_memory: bool = True) -> str:
        """Generate response using Groq API with RAG context"""
        try:
            if not self.groq_client:
                return "Error: Groq API key not configured. Please add your Groq API key to Streamlit secrets."
            
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

Always be concise but complete, and prioritize safety and accuracy."""

            user_prompt = f"""Question: {query}

Available Documentation Context:
{chr(10).join(context_docs) if context_docs else "No specific documentation found."}

{memory_context}

Please provide a helpful response based on the available information. If the documentation doesn't fully address the question, explain what you do know and suggest next steps."""

            # Generate response
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
            
            # Add to memory
            if use_memory:
                doc_sources = [result['metadata']['filename'] for result in self.search_engine.search(query, k=3)]
                self.memory_system.add_conversation(query, assistant_response, doc_sources)
            
            return assistant_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = ShifterRAGSystem()

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Main App
def main():
    st.title("🔧 Shifter Assistant RAG System (Simplified)")
    st.markdown("""
    *Intelligent assistance system for operational staff - Upload documentation and get instant, context-aware help*
    
    ⚠️ **Note**: This is a simplified version using basic text search instead of ML embeddings for better compatibility.
    """)
    
    # Sidebar
    st.sidebar.title("📚 Document Management")
    
    # Document upload section
    st.sidebar.header("Upload Documents")
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
                        doc_data = st.session_state.rag_system.doc_processor.process_document(uploaded_file)
                        if doc_data:
                            if st.session_state.rag_system.search_engine.add_document(doc_data):
                                success_count += 1
                    
                    if success_count > 0:
                        st.session_state.rag_system.save_search_index()
                        st.success(f"✅ Successfully processed {success_count} documents!")
                    else:
                        st.error("❌ No documents were processed successfully.")
    
    # Document statistics
    st.sidebar.header("📊 Document Library")
    total_docs = len(st.session_state.rag_system.search_engine.documents)
    total_chunks = len(st.session_state.rag_system.search_engine.chunks)
    
    st.sidebar.metric("Documents", total_docs)
    st.sidebar.metric("Text Chunks", total_chunks)
    
    if total_docs > 0:
        with st.sidebar.expander("📋 Document List"):
            for i, doc in enumerate(st.session_state.rag_system.search_engine.documents):
                st.write(f"**{doc['filename']}**")
                st.write(f"Type: {doc['file_type']} | Words: {doc['word_count']}")
                st.write(f"Uploaded: {doc['upload_time'][:10]}")
                st.write("---")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "🧠 Memory & Context", "⚙️ System Status"])
    
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
                with st.spinner("Searching documentation and generating response..."):
                    response = st.session_state.rag_system.generate_response(user_query, use_memory=use_memory)
                    
                    # Display response with better styling
                    st.markdown("### 🤖 Assistant Response")
                    st.markdown(f'<div class="response-container">{response}</div>', unsafe_allow_html=True)
                    
                    # Show relevant documents
                    search_results = st.session_state.rag_system.search_engine.search(user_query, k=3)
                    if search_results:
                        with st.expander("📚 Relevant Documentation"):
                            for i, result in enumerate(search_results):
                                st.write(f"**Source: {result['metadata']['filename']}** (Relevance: {result['score']:.3f})")
                                st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                                st.write("---")
        
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
                    with st.spinner("Getting information..."):
                        response = st.session_state.rag_system.generate_response(query)
                        st.markdown(f"**{query}**")
                        st.markdown(f'<div class="chat-message assistant-message">{response}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("🧠 Memory & Context Management")
        
        # Conversation history
        st.subheader("Recent Conversations")
        conversations = st.session_state.rag_system.memory_system.conversation_history[-10:]  # Last 10
        
        if conversations:
            for conv in reversed(conversations):
                with st.expander(f"Q: {conv['user_query'][:60]}... ({conv['timestamp'][:16]})"):
                    st.markdown(f"**Question:** {conv['user_query']}")
                    st.markdown(f"**Response:** {conv['assistant_response']}")
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
                    st.session_state.rag_system.memory_system.add_important_fact(new_fact, fact_category)
                    st.success("✅ Fact added to memory!")
        
        # Display important facts
        facts = st.session_state.rag_system.memory_system.important_facts
        if facts:
            for fact in reversed(facts[-10:]):  # Last 10 facts
                st.markdown(f'<div class="memory-item"><strong>{fact["category"].title()}:</strong> {fact["fact"]}<br><small>{fact["timestamp"][:16]}</small></div>', unsafe_allow_html=True)
        else:
            st.info("No important facts stored yet.")
        
        # Memory cleanup
        if st.button("🧹 Clear Conversation History"):
            st.session_state.rag_system.memory_system.conversation_history = []
            st.success("Conversation history cleared!")
    
    with tab3:
        st.header("⚙️ System Status")
        
        # API status
        st.subheader("🔌 API Configuration")
        if st.session_state.rag_system.groq_client:
            st.success("✅ Groq API: Connected")
        else:
            st.error("❌ Groq API: Not configured")
            st.info("Add your Groq API key to `.streamlit/secrets.toml`:")
            st.code('groq_api_key = "your_api_key_here"')
        
        # Search engine status
        st.subheader("🔍 Search Engine Status")
        st.info("ℹ️ Using simplified text search (no ML dependencies)")
        st.metric("Total Documents", len(st.session_state.rag_system.search_engine.documents))
        st.metric("Text Chunks", len(st.session_state.rag_system.search_engine.chunks))
        
        # Memory status
        st.subheader("🧠 Memory Status")
        st.metric("Conversations", len(st.session_state.rag_system.memory_system.conversation_history))
        st.metric("Important Facts", len(st.session_state.rag_system.memory_system.important_facts))
        
        # System actions
        st.subheader("🔧 System Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Search Index"):
                if st.session_state.rag_system.save_search_index():
                    st.success("✅ Search index saved!")
                else:
                    st.error("❌ Failed to save search index")
        
        with col2:
            if st.button("🔄 Reload Search Index"):
                st.session_state.rag_system.load_search_index()
                st.success("✅ Search index reloaded!")

if __name__ == "__main__":
    main() 