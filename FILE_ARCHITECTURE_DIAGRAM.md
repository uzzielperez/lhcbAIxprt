# 🏗️ File Architecture Diagram for shifter_rag_app_ace_vision.py

## 📁 **Core Application Files**

### **🎯 Main Application**
- **`shifter_rag_app_ace_vision.py`** - Main Streamlit app with ACE + Vision capabilities
- **`shifter_rag_app_vision.py`** - Base vision RAG app (copied from this)
- **`shifter_rag_app_ace.py`** - ACE-only version (earlier iteration)
- **`shifter_rag_app_simple.py`** - Simple RAG app (original base)

### **🧠 ACE Framework**
- **`ace_framework.py`** - Core ACE framework with all learning components
- **`ace_knowledge_graph.pkl`** - Persistent knowledge graph storage
- **`ace_demo.py`** - Demonstration of ACE capabilities
- **`test_ace_functionality.py`** - ACE system testing
- **`test_ace_setup.py`** - ACE setup verification

## 🔗 **Dependencies & Imports**

### **📚 Core Dependencies**
```python
# Main imports in shifter_rag_app_ace_vision.py
import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from ace_framework import ACESystem, FeedbackEntry  # 🧠 ACE Framework
```

### **🤖 AI/ML Libraries**
- **`groq`** - Groq API for LLaMA models
- **`openai`** - OpenAI API for GPT-4o vision
- **`anthropic`** - Anthropic API for Claude vision
- **`huggingface_vision_api.py`** - Hugging Face vision models
- **`huggingface_vision.py`** - Local Hugging Face models

### **📄 Document Processing**
- **`PyPDF2`** - PDF text extraction
- **`BeautifulSoup`** - HTML parsing
- **`PIL (Pillow)`** - Image processing

## 🗂️ **Data & Configuration Files**

### **📊 Persistent Data**
- **`shifter_docs_vision.pkl`** - Main document search index
- **`shifter_docs_simple.pkl`** - Simple document index
- **`ace_knowledge_graph.pkl`** - ACE knowledge graph storage
- **`test_ace_state_*.pkl`** - Test state files

### **⚙️ Configuration**
- **`.streamlit/secrets.toml`** - API keys and secrets
- **`requirements_ace.txt`** - ACE system dependencies
- **`requirements_rag_vision.txt`** - Vision RAG dependencies
- **`requirements_rag.txt`** - Basic RAG dependencies

## 📚 **Documentation Files**

### **📖 User Guides**
- **`README.md`** - Main project documentation
- **`ACE_README.md`** - ACE framework documentation
- **`ACE_INTEGRATION_GUIDE.md`** - Integration guide
- **`ACE_TESTING_GUIDE.md`** - Testing procedures
- **`CONVERSATIONAL_ACE_GUIDE.md`** - Conversational features
- **`QUERY_PROCESSING_FLOW.md`** - Query processing diagram

### **🔧 Technical Documentation**
- **`ACE_IMPLEMENTATION_SUMMARY.md`** - Implementation details
- **`4_CELLS_PROBLEM_FIX.md`** - 4-cells problem fix
- **`ACE_PERSISTENCE_FIX.md`** - Persistence fix
- **`STREAMLIT_FORM_FIX.md`** - Streamlit form fixes
- **`VISION_RAG_README.md`** - Vision RAG documentation
- **`SHIFTER_RAG_README.md`** - Shifter RAG documentation

## 🧪 **Testing & Development Files**

### **🧪 Test Scripts**
- **`test_ace_functionality.py`** - ACE functionality tests
- **`test_ace_setup.py`** - ACE setup tests
- **`test_rag_setup.py`** - RAG setup tests
- **`test_vision_setup.py`** - Vision setup tests
- **`test_hf_api.py`** - Hugging Face API tests

### **🛠️ Development Tools**
- **`debug_rag.py`** - RAG debugging
- **`setup_huggingface_vision.py`** - HF setup
- **`vision_demo_example.py`** - Vision demo
- **`vision_alternatives_guide.md`** - Vision alternatives

## 🔄 **File Relationships**

```
shifter_rag_app_ace_vision.py
├── ace_framework.py (🧠 ACE Framework)
├── huggingface_vision_api.py (🤗 HF Vision)
├── shifter_docs_vision.pkl (📚 Documents)
├── ace_knowledge_graph.pkl (🧠 Knowledge)
└── .streamlit/secrets.toml (⚙️ Config)

ace_framework.py
├── ContextEvolutionEngine
├── AdaptiveRAGPipeline
├── FeedbackCollector
└── ACEEvaluationSystem

huggingface_vision_api.py
├── HuggingFaceVisionProcessor
├── Model management
└── API integration
```

## 🎯 **Key File Dependencies**

### **Primary Dependencies:**
1. **`ace_framework.py`** - Core ACE learning system
2. **`huggingface_vision_api.py`** - Vision model integration
3. **`shifter_docs_vision.pkl`** - Document search index
4. **`.streamlit/secrets.toml`** - API configuration

### **Secondary Dependencies:**
1. **`shifter_rag_app_vision.py`** - Base vision app (copied from)
2. **`requirements_ace.txt`** - Python dependencies
3. **`ace_knowledge_graph.pkl`** - Persistent knowledge storage

### **Development Dependencies:**
1. **`test_ace_functionality.py`** - Testing
2. **`ace_demo.py`** - Demonstrations
3. **Documentation files** - User guides

## 🚀 **File Usage Flow**

```
1. User runs: streamlit run shifter_rag_app_ace_vision.py
2. App loads: ace_framework.py (ACE system)
3. App loads: huggingface_vision_api.py (Vision models)
4. App loads: shifter_docs_vision.pkl (Documents)
5. App loads: ace_knowledge_graph.pkl (Knowledge)
6. App reads: .streamlit/secrets.toml (API keys)
7. App initializes: All components
8. User interacts: Upload docs, ask questions, provide feedback
9. App saves: ace_knowledge_graph.pkl (Learning)
```

## 📋 **File Categories Summary**

| Category | Files | Purpose |
|----------|--------|---------|
| **Main App** | `shifter_rag_app_ace_vision.py` | Primary application |
| **ACE Framework** | `ace_framework.py` | Learning system |
| **Vision Models** | `huggingface_vision_api.py` | Vision processing |
| **Data Storage** | `*.pkl` files | Persistent data |
| **Configuration** | `secrets.toml` | API keys |
| **Documentation** | `*.md` files | User guides |
| **Testing** | `test_*.py` | Development tools |
| **Dependencies** | `requirements_*.txt` | Python packages |
