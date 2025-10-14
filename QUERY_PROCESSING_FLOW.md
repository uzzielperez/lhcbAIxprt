# 🔍 Query Processing Flow Diagram

## Current ACE + RAG System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY INPUT                                      │
│  "What should I do about this 4-cells problem?" (with image)                   │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ACE FRAMEWORK LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 1. Expert Knowledge Retrieval                                           │    │
│  │    - Search knowledge graph for "4-cells" related expert knowledge     │    │
│  │    - Retrieve: "Focus on 4 consecutive vertical cells pattern"         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 2. Query Enhancement                                                    │    │
│  │    Original: "What should I do about this 4-cells problem?"            │    │
│  │    Enhanced: "What should I do about this 4-cells problem?             │    │
│  │              Expert Knowledge: Focus on 4 consecutive vertical cells"  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │ Enhanced Query + Image Data
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RAG SYSTEM LAYER                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 3. Document Search (SearchEngine)                                     │    │
│  │    - Search uploaded documents for "4-cells" related content          │    │
│  │    - Returns: Top 3 most relevant document chunks                     │    │
│  │    - Example: "4-cells problem procedures from manual.pdf"          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 4. Memory Context (MemorySystem)                                      │    │
│  │    - Retrieve conversation history                                   │    │
│  │    - Get important facts about 4-cells problems                     │    │
│  │    - Add previous image analysis context                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │ Context: Documents + Memory + Expert Knowledge
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LLM API LAYER                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 5. Prompt Construction                                                 │    │
│  │    System: "You are an expert shifter assistant..."                   │    │
│  │    User: "Question: What should I do about this 4-cells problem?      │    │
│  │           Available Documentation Context: [RAG docs]                │    │
│  │           Memory Context: [conversation history]                      │    │
│  │           Expert Knowledge: Focus on 4 consecutive vertical cells"    │    │
│  │    Image: [Base64 encoded image data]                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 6. API Call to LLM                                                    │    │
│  │    - OpenAI GPT-4o-mini (Vision)                                      │    │
│  │    - OR Anthropic Claude 3.5 Sonnet                                   │    │
│  │    - OR Groq LLaMA 3.1 8B                                             │    │
│  │    - LLM processes: Query + Image + RAG docs + Memory + Expert knowledge │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │ LLM Response
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE PROCESSING                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 7. Response Generation                                                 │    │
│  │    LLM generates response using ALL available context:                │    │
│  │    - RAG documents (procedures, manuals)                             │    │
│  │    - Memory context (previous conversations)                         │    │
│  │    - Expert knowledge (your feedback)                                │    │
│  │    - Image analysis (visual understanding)                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 8. Learning Integration                                                │    │
│  │    - Store response in conversation history                          │    │
│  │    - Update knowledge graph with new concepts                        │    │
│  │    - Create relationships between visual and text knowledge          │    │
│  │    - Update learning metrics                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │ Final Response
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER RECEIVES                                         │
│  "Based on the 4-cells problem pattern in your image, you should:              │
│   1. Focus on the 4 consecutive vertical cells (as per expert knowledge)      │
│   2. Follow the restart procedure from the manual (from RAG docs)             │
│   3. Check the cooling system first (from memory context)                    │
│   4. Report to ProblemDQ with the affected run number"                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔍 **Key Components Analysis**

### **✅ What IS Working:**
1. **RAG Document Search**: ✅ Searches uploaded documents
2. **Memory Context**: ✅ Uses conversation history
3. **Expert Knowledge**: ✅ Now retrieves and uses stored expertise
4. **Image Analysis**: ✅ Processes uploaded images
5. **LLM Integration**: ✅ Sends all context to LLM

### **❌ What's NOT Working:**
1. **Knowledge Graph Integration**: The RAG system doesn't directly query the knowledge graph
2. **Document-Expert Connection**: No direct connection between RAG docs and expert knowledge
3. **Learning from Responses**: System doesn't learn from LLM responses

## 🧠 **How OpenAI Actually Processes Queries**

### **Current Flow:**
```
User Query → ACE Enhancement → RAG Search → Memory Context → LLM Prompt → Response
```

### **What OpenAI Sees:**
```
System: "You are an expert shifter assistant..."
User: "Question: What should I do about this 4-cells problem?
       Available Documentation Context: [RAG search results]
       Memory Context: [conversation history]
       Expert Knowledge: Focus on 4 consecutive vertical cells pattern"
Image: [Base64 image data]
```

### **OpenAI's Processing:**
1. **Analyzes the image** for 4-cells patterns
2. **Reads the RAG documents** for procedures
3. **Considers memory context** from previous conversations
4. **Applies expert knowledge** about 4-cells patterns
5. **Generates response** combining all sources

## 🎯 **The Answer to Your Question**

**YES, OpenAI IS going through:**
- ✅ **RAG Documents**: Searched and included in prompt
- ✅ **Knowledge Graph**: Expert knowledge retrieved and included
- ✅ **Memory Context**: Previous conversations included
- ✅ **Image Analysis**: Visual understanding applied

**The system works by:**
1. **Enhancing your query** with expert knowledge
2. **Searching RAG documents** for relevant procedures
3. **Including memory context** from previous conversations
4. **Sending everything to OpenAI** in a structured prompt
5. **OpenAI processes all sources** to generate the response

**The learning happens in the ACE layer, not in the LLM itself!** 🧠⚡
