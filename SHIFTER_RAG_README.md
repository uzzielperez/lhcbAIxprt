# 🔧 Shifter Assistant RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with Streamlit and Groq API to assist operational staff (shifters) with real-time, context-aware help based on uploaded documentation.

## ✨ Features

### 📚 Document Processing
- **Multi-format support**: PDF, HTML, TXT, Markdown files
- **Intelligent text extraction**: Clean text extraction from various document formats
- **Automatic chunking**: Smart text segmentation for optimal retrieval
- **Vector embeddings**: Uses sentence-transformers for semantic search

### 🔍 Advanced RAG Capabilities
- **Semantic search**: Find relevant information using natural language queries
- **Context-aware responses**: LLM responses enriched with relevant documentation
- **Source attribution**: Track which documents provided the information
- **Real-time processing**: Instant responses to shifter questions

### 🧠 Memory System
- **Conversation history**: Remembers previous interactions for better context
- **Important facts**: Store and recall critical information
- **Session context**: Maintains context within conversation sessions
- **Smart retrieval**: Uses conversation history to improve responses

### 🎯 Shifter-Specific Features
- **Quick actions**: Pre-defined common queries for fast access
- **Emergency procedures**: Prioritized access to critical information
- **Step-by-step guidance**: Clear, actionable instructions
- **Safety focus**: Highlights important safety considerations

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_rag.txt
```

### 2. Configure Groq API
1. Get your API key from [Groq Console](https://console.groq.com/keys)
2. Edit `.streamlit/secrets.toml`:
```toml
groq_api_key = "your_actual_groq_api_key_here"
```

### 3. Run the Application
```bash
streamlit run shifter_rag_app.py
```

## 📖 Usage Guide

### Document Management
1. **Upload Documents**: Use the sidebar to upload PDF, HTML, TXT, or Markdown files
2. **Process Documents**: Click "Process Documents" to add them to the knowledge base
3. **View Library**: Check document statistics and uploaded files in the sidebar

### Asking Questions
1. **Natural Language**: Ask questions in plain English
   - "What should I do if the cooling system alarm goes off?"
   - "How do I restart the data acquisition system?"
   - "Who should I contact for hardware issues?"

2. **Quick Actions**: Use pre-defined buttons for common queries
3. **Memory Context**: Enable conversation memory for better context

### Memory Management
1. **View History**: Check previous conversations in the Memory tab
2. **Add Facts**: Store important information that should be remembered
3. **Categorize**: Organize facts by type (safety, procedures, contacts, etc.)

## 🏗️ Architecture

### Core Components

1. **DocumentProcessor**: Handles file ingestion and text extraction
2. **VectorStore**: Manages embeddings and similarity search using FAISS
3. **MemorySystem**: Handles conversation history and important facts
4. **ShifterRAGSystem**: Main orchestrator integrating all components

### Data Flow
```
Upload Documents → Text Extraction → Chunking → Embeddings → Vector Store
                                                                    ↓
User Query → Semantic Search → Retrieve Relevant Chunks → LLM + Context → Response
            ↗                                                              ↓
Memory System ← Add to History ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

## 🧠 Adding Memory Features

The application includes several memory mechanisms:

### 1. Conversation Memory
**Automatic**: Every interaction is stored with:
- User query
- Assistant response
- Source documents used
- Timestamp
- Session ID

**Usage in code**:
```python
# Add conversation to memory
memory_system.add_conversation(user_query, assistant_response, doc_sources)

# Retrieve relevant history
relevant_history = memory_system.get_relevant_history(current_query, limit=3)
```

### 2. Important Facts Storage
**Manual**: Users can add critical information that should be remembered:
```python
# Add important fact
memory_system.add_important_fact(
    fact="Always check pressure readings before starting pump A",
    category="safety"
)
```

### 3. Session Context
**Automatic**: Maintains context within conversation sessions:
- Session-specific conversation tracking
- Context carryover between queries
- User preference learning

### 4. Enhancing Memory (Advanced Options)

#### A. Semantic Memory Search
Replace keyword matching with embedding-based memory retrieval:
```python
def get_relevant_history_semantic(self, query: str, limit: int = 3):
    # Encode conversation history
    conv_texts = [f"{conv['user_query']} {conv['assistant_response']}" 
                  for conv in self.conversation_history]
    conv_embeddings = self.model.encode(conv_texts)
    
    # Search similar conversations
    query_embedding = self.model.encode([query])
    similarities = cosine_similarity(query_embedding, conv_embeddings)[0]
    
    # Return top matches
    top_indices = similarities.argsort()[-limit:][::-1]
    return [self.conversation_history[i] for i in top_indices]
```

#### B. Automatic Fact Extraction
Extract important facts from conversations automatically:
```python
def extract_important_facts(self, conversation: str):
    prompt = f"""
    Extract important operational facts from this conversation that should be remembered:
    {conversation}
    
    Return facts as JSON list with category and importance score.
    """
    # Use LLM to extract facts automatically
```

#### C. User Profile Learning
Learn user preferences and expertise level:
```python
class UserProfile:
    def __init__(self):
        self.expertise_level = "beginner"  # beginner, intermediate, expert
        self.preferred_response_style = "detailed"  # concise, detailed, technical
        self.common_tasks = []
        self.frequent_questions = []
    
    def update_from_interactions(self, interactions):
        # Analyze interaction patterns to update profile
        pass
```

#### D. Contextual Memory Injection
Inject relevant memory into prompts intelligently:
```python
def build_context_prompt(self, query: str, search_results: List[str]):
    # Get relevant memory
    relevant_history = self.get_relevant_history(query)
    important_facts = self.get_relevant_facts(query)
    
    context = f"""
    Current Query: {query}
    
    Relevant Documentation:
    {chr(10).join(search_results)}
    
    Relevant Previous Discussions:
    {self.format_conversation_history(relevant_history)}
    
    Important Facts to Remember:
    {self.format_important_facts(important_facts)}
    """
    return context
```

## 🔧 Configuration Options

### Vector Store Settings
```python
# In VectorStore class
chunk_size = 1000      # Size of text chunks
overlap = 200          # Overlap between chunks
model_name = "all-MiniLM-L6-v2"  # Embedding model
```

### Memory Settings
```python
# In MemorySystem class
max_conversations = 50  # Maximum conversations to keep
max_facts = 100        # Maximum important facts to store
```

### LLM Settings
```python
# In generate_response method
model = "llama-3.1-70b-versatile"  # Groq model
temperature = 0.3                   # Response creativity
max_tokens = 1000                   # Response length
```

## 🛠️ Customization

### Adding New Document Types
```python
def process_custom_format(self, file_content: bytes) -> str:
    # Add custom processing logic
    return extracted_text

# Add to supported_formats list
self.supported_formats.append('.your_format')
```

### Custom Memory Categories
Update the category list in the UI:
```python
fact_category = st.selectbox("Category", [
    "general", "safety", "procedures", "contacts", 
    "troubleshooting", "your_custom_category"
])
```

### Enhanced Search
Add filters, date ranges, or document type filtering to search functionality.

## 🎯 Best Practices for Shifters

1. **Upload Comprehensive Documentation**: Include all relevant manuals, procedures, and guides
2. **Use Descriptive Filenames**: Help with source attribution and organization
3. **Regular Updates**: Keep documentation current and remove outdated information
4. **Memory Maintenance**: Regularly review and update important facts
5. **Test Responses**: Verify critical procedures with actual documentation

## 🔒 Security Considerations

- API keys stored in Streamlit secrets (not in code)
- Local vector store (no external data transmission)
- Session-based memory isolation
- No persistent storage of sensitive information

## 📊 Performance Tips

- **Document Size**: Keep documents under 10MB for optimal processing
- **Query Length**: Concise queries often yield better results
- **Memory Usage**: Clear old conversations if memory usage becomes high
- **Vector Store**: Periodically rebuild for optimal performance

## 🔄 Maintenance

### Regular Tasks
1. **Update Dependencies**: Keep libraries current for security
2. **Monitor API Usage**: Track Groq API consumption
3. **Review Memory**: Clean up irrelevant stored facts
4. **Document Updates**: Replace outdated documentation

### Troubleshooting
- **No API Response**: Check Groq API key configuration
- **Poor Search Results**: Verify document quality and relevance
- **Memory Issues**: Clear conversation history or restart application
- **Processing Errors**: Check file formats and encoding

## 🤝 Contributing

To extend the system:
1. Fork the project
2. Add new features or improvements
3. Test thoroughly with realistic shifter scenarios
4. Submit pull request with documentation

## 📝 License

This project is designed for operational use. Ensure compliance with your organization's policies regarding AI assistance tools.

---

*Built with ❤️ for operational excellence and shifter empowerment* 