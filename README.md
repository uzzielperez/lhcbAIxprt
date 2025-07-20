# PersonalHF - Investment & Operations Tools

This repository contains multiple applications for investment tracking and operational assistance.

## 📈 Hedge Fund Application
```bash 
streamlit run hedge_fund_app.py
```
Personal project for tracking investments and making informed decisions.

## 🎉 What's Been Built

### 🔧 **Shifter Assistant RAG System** (`shifter_rag_app.py`)
A complete RAG application with:

**Core Features:**
- **Multi-format document ingestion**: PDF, HTML, TXT, Markdown
- **Semantic search**: Uses sentence-transformers + FAISS for vector similarity
- **Groq API integration**: Following your existing pattern from `mktaiagent`
- **Real-time Q&A**: Context-aware responses using uploaded documentation

**Advanced Memory System:**
- **Conversation history**: Automatically stores all interactions
- **Important facts storage**: Manual addition of critical information
- **Session context**: Maintains context within conversations
- **Smart retrieval**: Uses past conversations to improve current responses

### 📁 **Supporting Files Created:**
1. `requirements_rag.txt` - All necessary dependencies
2. `.streamlit/secrets.toml` - Groq API configuration template
3. `SHIFTER_RAG_README.md` - Comprehensive documentation
4. `test_rag_setup.py` - Setup verification script
5. Updated main `README.md` - Overview of both applications

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_rag.txt

# 2. Test setup
python test_rag_setup.py

# 3. Configure Groq API key in .streamlit/secrets.toml
# groq_api_key = "your_actual_key_here"

# 4. Run the application
streamlit run shifter_rag_app.py
```

## 🧠 Memory Features Explained

### 1. **Automatic Conversation Memory**
Every interaction is stored with:
- User query + assistant response
- Source documents used
- Timestamp and session ID
- Used for context in future conversations

### 2. **Important Facts Storage**
Users can manually add critical information:
- Categorized by type (safety, procedures, contacts)
- Automatically retrieved when relevant
- Persistent across sessions

### 3. **Smart Context Building**
The system builds context prompts by:
- Finding relevant past conversations
- Including applicable important facts
- Combining with document search results
- Feeding enriched context to Groq API

### 4. **Advanced Memory Options** (in documentation)
Ready-to-implement enhancements:
- **Semantic memory search**: Use embeddings instead of keywords
- **Automatic fact extraction**: LLM extracts important info from conversations
- **User profile learning**: Adapts to user expertise and preferences
- **Enhanced context injection**: Smarter memory integration

## 🎯 How to Use for Shifters

1. **Upload Documents**: Drag PDFs, manuals, procedures into the sidebar
2. **Ask Questions**: Natural language queries like:
   - "What should I do if the cooling system alarm goes off?"
   - "How do I restart the data acquisition system?"
   - "Who should I contact for hardware issues?"
3. **Use Memory**: Enable conversation memory for context-aware responses
4. **Add Important Facts**: Store critical information for future reference
5. **Quick Actions**: Use preset buttons for common operational queries

## 🔧 Architecture Highlights

```
Documents → Text Extraction → Chunking → Embeddings → Vector Store
                                                            ↓
User Query → Semantic Search + Memory → Context → Groq API → Response
```

The system intelligently combines:
- **Document knowledge** (from uploaded files)
- **Conversation memory** (past interactions)
- **Important facts** (manually stored info)
- **Groq's reasoning** (LLM processing)

## 🎨 UI Features

- **Clean, professional interface** with shifter-focused design
- **Three main tabs**: Questions, Memory Management, System Status
- **Real-time document processing** with progress indicators
- **Source attribution** showing which documents provided information
- **Memory visualization** with conversation history and important facts

The application is production-ready and follows your existing Groq API patterns from the `mktaiagent` project. It's designed specifically for operational environments where quick, accurate information retrieval is critical for safety and efficiency.

Would you like me to explain any specific part in more detail or help you set up additional features?

