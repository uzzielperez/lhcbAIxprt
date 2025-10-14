# 🔧 LHCb AI Expert - Shifter Assistant RAG System

An intelligent RAG (Retrieval-Augmented Generation) system designed to assist LHCb shifters with real-time, context-aware help based on uploaded documentation and procedures.

The noob (in this case me!) can do the following: 
- **Upload twiki html files and some relevant e-logs** and process the documents
- With the processed documents, the shifter can quickly retrieve the relevant answers to their questions and save time scouring through documents. So far, only the "CALO/PLUME documents have been tried. 

Sample questions: 
- Tell me the PLUME Piquet checklist
- How to solve the 4 vertical cells problem?
- How do I restart either the ECAL or the HCAL system?
- Give me the numbers of the CALO and PLUME experts

OK! Now just need to stress test this. 

## 🧠 Agentic Context Engineering (ACE) Framework

Shift operations at CERN's LHCb experiment are challenged by vast, dynamically evolving documentation, which hinders rapid information access for novice shifters and risks operational errors. We introduce **Agentic Context Engineering (ACE)**, a novel framework that enables large language models (LLMs) to autonomously self-improve by iteratively evolving contextual knowledge and refining their Retrieval-Augmented Generation (RAG) pipelines. 

Leveraging open-source models like LLaMA 3.x hosted on Groq for efficient inference, ACE dynamically retrieves, summarizes, and updates shift documentation based on real-time human expert feedback. A multimodal extension, powered by GPT-4o, incorporates vision capabilities to analyze plots, dashboards, and visual data for enhanced reasoning.

Through this integration of agentic context evolution and adaptive RAG mechanisms, ACE delivers a reliable, trustworthy AI assistant that continuously adapts to LHCb's operational needs, accelerating decision-making, reducing training overhead, and improving overall experiment reliability. Future evaluations will assess its impact on shift efficiency and error rates in live deployments.

## ✨ Features

### 📚 **Document Processing**
- **Multi-format support**: PDF, HTML, TXT, Markdown files
- **Smart encoding detection**: Handles international characters and legacy file formats
- **Automatic text extraction**: Clean text extraction from various document formats
- **Intelligent chunking**: Optimized text segmentation for better retrieval

### 🔍 **Advanced Search & RAG**
- **Text-based search**: Fast, reliable keyword-based document retrieval
- **Context-aware responses**: LLM responses enriched with relevant documentation
- **Source attribution**: Track which documents provided the information
- **Real-time processing**: Instant responses to shifter questions

### 🧠 **Memory System**
- **Conversation history**: Remembers previous interactions for better context
- **Important facts storage**: Manual addition of critical information
- **Session context**: Maintains context within conversation sessions
- **Smart retrieval**: Uses conversation history to improve responses

### 🎯 **Shifter-Specific Features**
- **Quick actions**: Pre-defined common queries for fast access
- **Emergency procedures**: Prioritized access to critical information
- **Step-by-step guidance**: Clear, actionable instructions
- **Safety focus**: Highlights important safety considerations

## 🚀 Setup Instructions

### 1. **Install Dependencies**

First, install the required packages using conda (recommended for compatibility):

```bash
# Core dependencies
conda install -c conda-forge streamlit groq pandas numpy pypdf2 beautifulsoup4 -y

# Additional packages via pip if needed
pip install groq
```

**Alternative using pip:**
```bash
pip install -r requirements_rag.txt
```

### 2. **Configure Groq API**

1. Get your API key from [Groq Console](https://console.groq.com/keys)
2. Create/edit `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
groq_api_key = "gsk_your_actual_groq_api_key_here"
```

⚠️ **Important**: Replace `"gsk_your_actual_groq_api_key_here"` with your actual Groq API key.

### 3. **Test Your Setup**

Run the diagnostic script to verify everything is working:

```bash
python test_rag_setup_simple.py
```

Expected output:
```
✅ Core modules: 12/12 working
✅ FAISS - OK
✅ Groq API connection successful
```

### 4. **Start the Application**

```bash
streamlit run shifter_rag_app_simple.py
```

The application will be available at: **http://localhost:8501**

## 📖 Usage Guide

### **Document Management**
1. **Upload Documents**: Use the sidebar to upload PDF, HTML, TXT, or Markdown files
2. **Process Documents**: Click "Process Documents" to add them to the knowledge base
3. **View Library**: Check document statistics and uploaded files in the sidebar

### **Asking Questions**
1. **Natural Language**: Ask questions in plain English:
   - "What should I check during my shift?"
   - "How do I handle a detector alarm?"
   - "What are the emergency procedures?"

2. **Quick Actions**: Use pre-defined buttons for common queries
3. **Memory Context**: Enable conversation memory for better context (recommended)
4. **Enter Key**: Press Enter or click "Get Help" to submit questions

### **Memory Management**
1. **View History**: Check previous conversations in the Memory tab
2. **Add Facts**: Store important information that should be remembered
3. **Categorize**: Organize facts by type (safety, procedures, contacts, etc.)

## 🏗️ System Architecture

### **Files Overview**
- `shifter_rag_app_simple.py` - **Main application** (use this one)
- `shifter_rag_app.py` - Advanced version with ML embeddings (may have compatibility issues)
- `debug_rag.py` - Diagnostic script for troubleshooting
- `test_rag_setup_simple.py` - Setup verification
- `.streamlit/secrets.toml` - API key configuration
- `requirements_rag.txt` - Python dependencies

### **Data Flow**
```
Upload Documents → Text Extraction → Chunking → Text Search Index
                                                        ↓
User Query → Keyword Search → Retrieve Relevant Chunks → Groq API + Context → Response
            ↗                                                              ↓
Memory System ← Add to History ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

## 🔧 Technical Details

### **Search Engine**
- **Type**: Text-based keyword matching with Jaccard similarity
- **Advantages**: Fast, reliable, no ML dependencies
- **Performance**: Instant search results
- **Encoding**: Handles multiple character encodings (UTF-8, Latin-1, Windows-1252, etc.)

### **LLM Integration**
- **Model**: `llama-3.1-8b-instant` (via Groq API)
- **Context**: Combines document chunks + conversation memory
- **Safety**: Prioritizes accuracy and operational safety

### **Memory System**
- **Conversation Storage**: Last 50 interactions
- **Important Facts**: Categorized critical information
- **Context Building**: Smart retrieval for enhanced responses

## 🛠️ Troubleshooting

### **Common Issues**

#### "File does not exist" Error
```bash
# Make sure you're in the correct directory
cd /path/to/lhcbAIxprt
pwd  # Should show: /Users/uzzielperez/Desktop/lhcbAIxprt
ls shifter_rag_app_simple.py  # Should show the file
```

#### Encoding Errors (e.g., 'utf-8' codec can't decode)
✅ **Fixed**: The app now handles multiple encodings automatically
- Supports UTF-8, Latin-1, Windows-1252, ISO-8859-1, CP1252
- Gracefully handles problematic characters

#### No Response from Queries
1. Check Groq API key configuration
2. Verify documents are uploaded and processed
3. Try simple queries first (words that appear in your documents)

#### Dependencies Issues
```bash
# For conda environments
conda install -c conda-forge pypdf2 beautifulsoup4 -y

# Test basic functionality
python debug_rag.py
```

### **Debug Commands**

```bash
# Check system status
python debug_rag.py

# Test core dependencies
python test_rag_setup_simple.py

# View uploaded documents
ls -la *.pkl  # Should show shifter_docs_simple.pkl
```

## 📊 Performance Tips

- **Document Size**: Keep documents under 10MB for optimal processing
- **Query Style**: Use descriptive queries that match document content
- **Memory Usage**: Clear old conversations if memory usage becomes high
- **File Formats**: HTML and PDF work best; ensure good text content quality

## 🔒 Security & Privacy

- **API Keys**: Stored locally in Streamlit secrets (not in code)
- **Local Processing**: Documents processed and stored locally
- **No External Data**: No document content sent to external services except Groq for responses
- **Session Isolation**: Memory is session-based

## 🎯 Best Practices for Shifters

1. **Upload Comprehensive Documentation**: Include all relevant procedures, manuals, and guides
2. **Use Descriptive Filenames**: Helps with source attribution and organization
3. **Regular Updates**: Keep documentation current and remove outdated information
4. **Memory Maintenance**: Add important facts and review stored information
5. **Test Responses**: Verify critical procedures with actual documentation

## 📝 Example Queries

### **For PLUME System:**
- "What should I check during my piquet shift?"
- "How do I handle PLUME alarms?"
- "What are the startup procedures?"

### **For CALO System:**
- "CALO detector troubleshooting steps"
- "Emergency shutdown procedures for CALO"
- "Who do I contact for CALO issues?"

### **General Operations:**
- "Daily inspection checklist"
- "Emergency contact information"
- "System restart procedures"

## 🤝 Contributing

To extend or improve the system:
1. Test changes with realistic shifter scenarios
2. Ensure backward compatibility
3. Update documentation
4. Verify encoding handling for international content

## 📞 Support

For issues or improvements:
1. Check the troubleshooting section above
2. Run `python debug_rag.py` for diagnostic information
3. Verify your setup with `python test_rag_setup_simple.py`

---

*Built for LHCb operational excellence and shifter empowerment* 🔬⚡

