# 🔧👁️ LHCb AI Expert - Shifter Assistant RAG System with Vision

An intelligent RAG (Retrieval-Augmented Generation) system with **vision capabilities** designed to assist LHCb shifters with real-time, context-aware help based on uploaded documentation AND visual analysis of images, diagrams, error screens, and equipment photos.

## 🆕 What's New: Vision Capabilities

The enhanced system now includes:
- **🖼️ Image Analysis**: Upload photos of equipment, error screens, diagrams, and get AI-powered visual analysis
- **👁️ Multimodal Understanding**: Combine text documentation with visual information for comprehensive assistance
- **📸 Image Memory**: Track and reference previous image analyses in conversations
- **🤖 Advanced Vision Models**: Support for LLaMA 3.2 Vision, LLaVA, and other multimodal models via Groq API

## ✨ Features

### 📚 **Document Processing** (Enhanced)
- **Multi-format support**: PDF, HTML, TXT, Markdown files
- **Smart encoding detection**: Handles international characters and legacy file formats
- **Automatic text extraction**: Clean text extraction from various document formats
- **Intelligent chunking**: Optimized text segmentation for better retrieval

### 👁️ **Vision Analysis** (NEW!)
- **Image Upload**: Support for JPG, PNG, GIF, BMP, WebP images (up to 20MB)
- **Visual Understanding**: AI-powered analysis of equipment photos, diagrams, error screens
- **Contextual Analysis**: Combines visual information with existing documentation
- **Image History**: Track and reference previous visual analyses

### 🔍 **Advanced Search & RAG**
- **Text-based search**: Fast, reliable keyword-based document retrieval
- **Vision-enhanced responses**: LLM responses enriched with both textual and visual context
- **Source attribution**: Track which documents and images provided the information
- **Real-time processing**: Instant responses to both text and image queries

### 🧠 **Enhanced Memory System**
- **Conversation history**: Remembers previous interactions including image analyses
- **Image analysis storage**: Maintains history of visual analyses and insights
- **Important facts storage**: Manual addition of critical information
- **Session context**: Maintains context within conversation sessions
- **Smart retrieval**: Uses conversation and image history to improve responses

### 🎯 **Shifter-Specific Features**
- **Visual troubleshooting**: Analyze equipment photos and error screens
- **Diagram interpretation**: Understand technical diagrams and schematics
- **Step-by-step guidance**: Clear, actionable instructions based on visual and textual context
- **Safety focus**: Highlights important safety considerations from both text and images

## 🚀 Setup Instructions

### 1. **Install Enhanced Dependencies**

Install the required packages including vision processing:

```bash
# Using pip (recommended)
pip install -r requirements_rag_vision.txt
```

**Manual installation:**
```bash
pip install streamlit groq pandas numpy PyPDF2 beautifulsoup4 Pillow
```

### 2. **Configure Groq API**

1. Get your API key from [Groq Console](https://console.groq.com/keys)
2. Create/edit `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
groq_api_key = "gsk_your_actual_groq_api_key_here"
```

⚠️ **Important**: Replace `"gsk_your_actual_groq_api_key_here"` with your actual Groq API key.

### 3. **Test Your Vision Setup**

Run the enhanced diagnostic script:

```bash
python test_vision_setup.py
```

Expected output:
```
✅ Core Dependencies: 13/13 working
✅ Vision Dependencies: 1/1 working
✅ Image Processing capabilities working
✅ Vision Models configured
✅ Groq API connection successful
🎉 All tests passed! Your vision RAG system is ready to use.
```

### 4. **Start the Enhanced Application**

```bash
streamlit run shifter_rag_app_vision.py
```

The application will be available at: **http://localhost:8501**

## 📖 Usage Guide

### **Document Management** (Enhanced)
1. **Upload Documents**: Use the sidebar to upload PDF, HTML, TXT, or Markdown files
2. **Process Documents**: Click "Process Documents" to add them to the knowledge base
3. **View Library**: Check document statistics and uploaded files in the sidebar

### **Vision Analysis** (NEW!)
1. **Upload Images**: Go to the "👁️ Vision Analysis" tab
2. **Select Image**: Upload photos of equipment, error screens, diagrams, etc.
3. **Ask Questions**: Describe what you want to know about the image
4. **Get AI Analysis**: Receive detailed visual analysis with actionable guidance
5. **Review History**: Check previous image analyses in the same tab

### **Enhanced Questioning**
1. **Text-only Questions**: Use the main "💬 Ask Questions" tab for traditional text queries
2. **Vision-enhanced Questions**: Upload images in the Vision tab for multimodal analysis
3. **Combined Context**: The system remembers both text and image conversations
4. **Memory Integration**: Previous image analyses inform future text responses

### **Model Selection**
- **Vision Models**: Choose from LLaMA 3.2 Vision (90B/11B) or LLaVA models
- **Text Models**: Select from LLaMA 3.1 variants or Mixtral for text-only queries
- **Automatic Switching**: System uses appropriate model based on query type

## 🏗️ Enhanced System Architecture

### **Files Overview**
- `shifter_rag_app_vision.py` - **Enhanced main application with vision** (use this one)
- `shifter_rag_app_simple.py` - Original text-only version
- `test_vision_setup.py` - Vision setup verification
- `requirements_rag_vision.txt` - Enhanced dependencies including vision
- `.streamlit/secrets.toml` - API key configuration

### **Enhanced Data Flow**
```
Upload Documents → Text Extraction → Chunking → Text Search Index
                                                        ↓
Upload Images → Image Processing → Base64 Encoding → Vision Model Analysis
                      ↓                                    ↓
User Query (Text/Vision) → Search + Vision Analysis → Groq API + Context → Response
                    ↗                                                        ↓
Memory System ← Add to History (Text + Images) ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

## 🔧 Technical Details

### **Vision Processing**
- **Image Formats**: JPG, PNG, GIF, BMP, WebP
- **Size Limits**: 20MB maximum per image
- **Processing**: Automatic resizing, format conversion, base64 encoding
- **Models**: LLaMA 3.2 Vision (90B/11B), LLaVA 1.5 7B

### **Enhanced LLM Integration**
- **Vision Models**: `llama-3.2-90b-vision-preview` (recommended), `llama-3.2-11b-vision-preview`, `llava-v1.5-7b-4096`
- **Text Models**: `llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`
- **Context**: Combines document chunks + image analysis + conversation memory
- **Multimodal**: Supports both text-only and vision-enhanced queries

### **Enhanced Memory System**
- **Conversation Storage**: Last 50 interactions (text + vision)
- **Image History**: Last 20 image analyses with metadata
- **Important Facts**: Categorized critical information
- **Context Building**: Smart retrieval for enhanced multimodal responses

## 🛠️ Troubleshooting

### **Vision-Specific Issues**

#### Image Upload Errors
```bash
# Check supported formats
Supported: JPG, JPEG, PNG, GIF, BMP, WebP
Max size: 20MB

# Test image processing
python test_vision_setup.py
```

#### Vision Model Errors
1. Verify Groq API key supports vision models
2. Check model availability in your region
3. Try different vision models from the dropdown

#### "Pillow not found" Error
```bash
pip install Pillow
# or
conda install pillow
```

### **General Issues**

#### Dependencies Issues
```bash
# Install all vision dependencies
pip install -r requirements_rag_vision.txt

# Test setup
python test_vision_setup.py
```

#### No Response from Vision Queries
1. Check Groq API key configuration
2. Verify image was processed successfully
3. Try different vision models
4. Ensure image is clear and relevant

## 📊 Performance Tips

- **Image Size**: Keep images under 5MB for faster processing
- **Image Quality**: Use clear, well-lit images for better analysis
- **Query Specificity**: Ask specific questions about images for better results
- **Model Selection**: Use LLaMA 3.2 90B for complex visual analysis, 11B for faster responses

## 🎯 Vision Use Cases for Shifters

### **Equipment Monitoring**
- Upload photos of control panels, displays, meters
- Get status interpretations and troubleshooting advice
- Compare current state with documented procedures

### **Error Analysis**
- Photograph error screens, alarms, warning lights
- Receive detailed error explanations and resolution steps
- Cross-reference with documentation automatically

### **Procedure Verification**
- Upload photos of equipment configurations
- Verify correct setup against documented procedures
- Get step-by-step guidance for corrections

### **Documentation Enhancement**
- Analyze technical diagrams and schematics
- Get explanations of complex visual information
- Combine visual and textual documentation

## 📝 Example Vision Queries

### **Equipment Analysis:**
- "What does this control panel display indicate?"
- "Is this equipment configuration correct according to procedures?"
- "What should I check based on what you see in this photo?"

### **Error Diagnosis:**
- "What error is shown on this screen and how do I fix it?"
- "Analyze this alarm display and suggest next steps"
- "Compare this error with documented troubleshooting procedures"

### **Procedure Verification:**
- "Does this setup match the installation diagram?"
- "What's wrong with this configuration based on the manual?"
- "Guide me through correcting what you see in this image"

## 🔒 Security & Privacy (Enhanced)

- **API Keys**: Stored locally in Streamlit secrets (not in code)
- **Local Processing**: Documents and images processed locally before API calls
- **Image Data**: Images converted to base64 and sent securely to Groq API
- **No Storage**: Images not permanently stored on external servers
- **Session Isolation**: Memory is session-based and temporary

## 🤝 Contributing

To extend or improve the vision system:
1. Test changes with realistic shifter scenarios including images
2. Ensure backward compatibility with text-only features
3. Update documentation for new vision capabilities
4. Verify image processing across different formats and sizes

## 📞 Support

For issues or improvements:
1. Check the troubleshooting section above
2. Run `python test_vision_setup.py` for diagnostic information
3. Test with different image formats and sizes
4. Verify Groq API key supports vision models

---

*Enhanced with AI Vision for LHCb operational excellence and shifter empowerment* 🔬⚡👁️
