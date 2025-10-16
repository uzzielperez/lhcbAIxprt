#!/usr/bin/env python3
"""
Image Memory Demo - Shows how the ACE system remembers and uses images
"""

import streamlit as st
from ace_framework import ACESystem
import tempfile
import os

def demo_image_memory():
    """Demonstrate how the system remembers images"""
    
    st.title("🖼️ Image Memory Demo")
    st.markdown("""
    This demo shows how the ACE system remembers images and uses them to provide context-aware responses.
    
    **How it works:**
    1. **Upload an image** - The system analyzes what it sees
    2. **Store visual concepts** - Extracts key information and stores it in the knowledge graph
    3. **Link to text knowledge** - Connects visual information with relevant documentation
    4. **Provide context-aware responses** - Uses both image analysis and RAG knowledge
    """)
    
    # Initialize ACE system
    if 'ace_system' not in st.session_state:
        st.session_state.ace_system = ACESystem(None)  # Demo mode
    
    # Image upload
    st.header("📸 Step 1: Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload an image of equipment, error screens, diagrams, etc."
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Create image data
        image_data = {
            'filename': uploaded_file.name,
            'dimensions': (800, 600),  # Demo dimensions
            'upload_time': '2025-01-16 18:40:00',
            'file_size': uploaded_file.size
        }
        
        st.success(f"✅ Image processed: {uploaded_file.name}")
        
        # Show what the system extracts
        st.header("🧠 Step 2: Visual Concept Extraction")
        
        # Extract visual concepts (demo)
        visual_concepts = [
            {
                "type": "image_metadata",
                "content": f"Image file: {uploaded_file.name}"
            },
            {
                "type": "image_properties", 
                "content": f"Image dimensions: 800x600"
            },
            {
                "type": "troubleshooting",
                "content": "Visual troubleshooting context for equipment analysis"
            },
            {
                "type": "equipment_analysis",
                "content": "Equipment analysis context for system monitoring"
            }
        ]
        
        st.write("**Visual concepts extracted:**")
        for i, concept in enumerate(visual_concepts, 1):
            st.write(f"{i}. **{concept['type']}**: {concept['content']}")
        
        # Show knowledge graph integration
        st.header("🔗 Step 3: Knowledge Graph Integration")
        
        st.write("**The system will:**")
        st.write("1. **Store visual concepts** in the knowledge graph")
        st.write("2. **Link to relevant documentation** from your RAG system")
        st.write("3. **Create relationships** between visual and text knowledge")
        st.write("4. **Enable context-aware responses** for future queries")
        
        # Demo query
        st.header("💬 Step 4: Context-Aware Query")
        
        demo_query = st.text_area(
            "Ask about the image:",
            value="Describe what you see and suggest the next steps",
            help="The system will use both the image analysis and your RAG knowledge"
        )
        
        if st.button("🔍 Analyze with Memory"):
            with st.spinner("🧠 ACE system analyzing with image memory..."):
                # Simulate the analysis
                st.markdown("### 👁️ Vision Analysis")
                st.write("**What I see:** Equipment monitoring display with status indicators")
                st.write("**Analysis:** System appears to be in normal operating mode")
                
                st.markdown("### 📚 RAG Knowledge Retrieved")
                st.write("**Relevant documentation found:**")
                st.write("- System monitoring procedures")
                st.write("- Troubleshooting guides")
                st.write("- Equipment maintenance schedules")
                
                st.markdown("### 🎯 Context-Aware Response")
                st.write("**Based on the image and documentation:**")
                st.write("1. **Current Status**: System appears normal")
                st.write("2. **Next Steps**: Continue monitoring, check logs if needed")
                st.write("3. **Recommendations**: Schedule routine maintenance")
                
                st.markdown("### 🧠 ACE Learning Applied")
                st.write("✅ **Visual concepts stored** in knowledge graph")
                st.write("✅ **Relationships created** between image and text knowledge")
                st.write("✅ **Context enhanced** with expert knowledge")
                st.write("✅ **System improved** for future similar queries")
    
    # Show knowledge graph status
    st.header("📊 Knowledge Graph Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", "65", "+2")
    with col2:
        st.metric("Visual Nodes", "8", "+4")
    with col3:
        st.metric("Relationships", "142", "+6")
    
    st.info("""
    **🎯 Key Benefits:**
    - **Persistent Memory**: Images are remembered across sessions
    - **Context Linking**: Visual information connects to relevant documentation
    - **Improved Responses**: Future queries use both image analysis and RAG knowledge
    - **Continuous Learning**: System gets better with each image analyzed
    """)

if __name__ == "__main__":
    demo_image_memory()
