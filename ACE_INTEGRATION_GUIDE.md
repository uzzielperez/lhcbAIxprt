# 🧠 ACE Integration Guide for LHCb RAG System

## Overview

This guide explains how to integrate the **Agentic Context Engineering (ACE)** framework with your existing LHCb Shifter Assistant RAG system. ACE transforms your static RAG system into an autonomous, self-improving AI assistant.

## 🚀 Quick Integration

### Step 1: Install ACE Framework

```bash
# Install ACE dependencies
pip install -r requirements_ace.txt

# Or install specific packages
pip install streamlit pandas numpy groq pypdf2 beautifulsoup4 matplotlib
```

### Step 2: Add ACE to Existing System

Replace your existing RAG system initialization:

```python
# Before (existing system)
from shifter_rag_app_simple import ShifterRAGSystem
rag_system = ShifterRAGSystem()

# After (with ACE)
from ace_framework import ACESystem
from shifter_rag_app_simple import ShifterRAGSystem

base_rag = ShifterRAGSystem()
ace_system = ACESystem(base_rag)
```

### Step 3: Update Query Processing

Replace your query processing:

```python
# Before
response = rag_system.generate_response(query)

# After (with ACE)
response, ace_metrics = ace_system.process_query_with_ace(query)
```

### Step 4: Add Feedback Collection

Add feedback collection to your interface:

```python
# Collect user feedback
feedback_result = ace_system.collect_feedback(
    query=user_query,
    response=system_response,
    user_rating=rating,  # 1-5 scale
    expert_correction=correction,  # Optional
    improvement_suggestions=suggestions  # Optional
)
```

## 🔧 Detailed Integration

### 1. Minimal Integration

For the simplest integration, just wrap your existing RAG system:

```python
import streamlit as st
from ace_framework import ACESystem
from shifter_rag_app_simple import ShifterRAGSystem

# Initialize ACE system
if 'ace_system' not in st.session_state:
    base_rag = ShifterRAGSystem()
    st.session_state.ace_system = ACESystem(base_rag)

# Use ACE for queries
def get_ace_response(query):
    response, metrics = st.session_state.ace_system.process_query_with_ace(query)
    return response, metrics

# Add feedback collection
def collect_feedback(query, response, rating):
    return st.session_state.ace_system.collect_feedback(
        query, response, rating
    )
```

### 2. Full Integration with UI

Replace your main application with ACE-enhanced version:

```python
# In your main app file
import streamlit as st
from ace_framework import ACESystem
from shifter_rag_app_simple import ShifterRAGSystem

# Initialize ACE system
if 'ace_system' not in st.session_state:
    base_rag = ShifterRAGSystem()
    st.session_state.ace_system = ACESystem(base_rag)
    st.session_state.ace_system.load_ace_state("shifter_docs_ace")

# Main query interface
def main():
    st.title("🧠 LHCb Shifter Assistant with ACE")
    
    # Query input
    query = st.text_area("Ask a question:")
    
    if st.button("Get ACE Response"):
        with st.spinner("🧠 ACE system processing..."):
            response, metrics = st.session_state.ace_system.process_query_with_ace(query)
            
            st.markdown("### Response")
            st.write(response)
            
            # Show ACE metrics
            if metrics["ace_applied"]:
                st.success("🧠 ACE enhancement applied!")
            
            st.metric("Context Enhancement", f"{metrics['context_enhancement_ratio']:.2f}x")
    
    # Feedback collection
    st.subheader("Rate this response:")
    rating = st.slider("Rating", 1, 5, 3)
    
    if st.button("Submit Feedback"):
        feedback_result = st.session_state.ace_system.collect_feedback(
            query, response, rating
        )
        st.success("Feedback submitted!")
    
    # Save ACE state
    if st.button("Save ACE State"):
        st.session_state.ace_system.save_ace_state("shifter_docs_ace")
        st.success("ACE state saved!")
```

### 3. Advanced Integration with Metrics

For full ACE capabilities, integrate all components:

```python
import streamlit as st
from ace_framework import ACESystem
from shifter_rag_app_simple import ShifterRAGSystem

def main():
    st.title("🧠 LHCb Shifter Assistant with ACE")
    
    # Initialize ACE system
    if 'ace_system' not in st.session_state:
        base_rag = ShifterRAGSystem()
        st.session_state.ace_system = ACESystem(base_rag)
        st.session_state.ace_system.load_ace_state("shifter_docs_ace")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Query Interface", 
        "🧠 Knowledge Graph", 
        "📊 Analytics", 
        "🎯 Feedback"
    ])
    
    with tab1:
        # Query interface with ACE
        query = st.text_area("Ask a question:")
        
        if st.button("Get ACE Response"):
            response, metrics = st.session_state.ace_system.process_query_with_ace(query)
            st.write(response)
            
            # Show ACE metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Base Contexts", metrics["base_contexts_count"])
            with col2:
                st.metric("Evolved Contexts", metrics["evolved_contexts_count"])
            with col3:
                st.metric("Enhancement", f"{metrics['context_enhancement_ratio']:.2f}x")
    
    with tab2:
        # Knowledge graph visualization
        ace_metrics = st.session_state.ace_system.get_ace_metrics()
        
        st.metric("Knowledge Nodes", ace_metrics["knowledge_graph_size"])
        st.metric("Relationships", ace_metrics["total_relationships"])
        
        # Display knowledge nodes
        context_engine = st.session_state.ace_system.context_engine
        for node_id, node in context_engine.knowledge_graph.items():
            with st.expander(f"{node.category} (confidence: {node.confidence:.2f})"):
                st.write(node.content)
    
    with tab3:
        # Learning analytics
        learning_metrics = ace_metrics["learning_metrics"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interactions", learning_metrics["total_interactions"])
        with col2:
            st.metric("Positive Feedback", learning_metrics["positive_feedback"])
        with col3:
            st.metric("Accuracy", f"{learning_metrics['accuracy_improvement']:.2%}")
    
    with tab4:
        # Feedback collection
        st.subheader("Provide Feedback")
        
        feedback_query = st.text_area("Query:")
        feedback_response = st.text_area("Response:")
        rating = st.slider("Rating", 1, 5, 3)
        expert_correction = st.text_area("Expert Correction (optional):")
        
        if st.button("Submit Feedback"):
            feedback_result = st.session_state.ace_system.collect_feedback(
                feedback_query, feedback_response, rating, expert_correction
            )
            st.success("Feedback submitted!")
            st.json(feedback_result)
```

## 🔄 Migration Strategies

### Strategy 1: Gradual Migration

1. **Phase 1**: Add ACE alongside existing system
2. **Phase 2**: Route some queries through ACE
3. **Phase 3**: Fully migrate to ACE system

```python
# Phase 1: Side-by-side comparison
def get_response_with_comparison(query):
    # Original system
    original_response = original_rag.generate_response(query)
    
    # ACE system
    ace_response, metrics = ace_system.process_query_with_ace(query)
    
    return {
        "original": original_response,
        "ace": ace_response,
        "metrics": metrics
    }
```

### Strategy 2: A/B Testing

```python
import random

def get_response_with_ab_testing(query):
    # Randomly choose between original and ACE
    use_ace = random.random() < 0.5  # 50% chance
    
    if use_ace:
        response, metrics = ace_system.process_query_with_ace(query)
        return response, {"system": "ace", "metrics": metrics}
    else:
        response = original_rag.generate_response(query)
        return response, {"system": "original"}
```

### Strategy 3: Full Replacement

```python
# Direct replacement
class EnhancedRAGSystem:
    def __init__(self):
        self.base_rag = ShifterRAGSystem()
        self.ace_system = ACESystem(self.base_rag)
    
    def generate_response(self, query, use_memory=True):
        response, metrics = self.ace_system.process_query_with_ace(query, use_memory)
        return response
    
    def collect_feedback(self, query, response, rating, correction=None):
        return self.ace_system.collect_feedback(query, response, rating, correction)
```

## 📊 Monitoring and Evaluation

### Key Metrics to Track

1. **Learning Metrics**
   - Total interactions
   - Positive feedback rate
   - Accuracy improvement over time

2. **Performance Metrics**
   - Response quality scores
   - Context enhancement ratios
   - Adaptation frequency

3. **Knowledge Graph Metrics**
   - Node count and growth
   - Relationship density
   - Confidence score distribution

### Monitoring Dashboard

```python
def create_monitoring_dashboard():
    ace_metrics = ace_system.get_ace_metrics()
    
    # Learning progress
    st.subheader("Learning Progress")
    learning_metrics = ace_metrics["learning_metrics"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Interactions", learning_metrics["total_interactions"])
    with col2:
        st.metric("Accuracy", f"{learning_metrics['accuracy_improvement']:.2%}")
    with col3:
        st.metric("Quality Score", f"{learning_metrics['response_quality_score']:.3f}")
    
    # Knowledge graph stats
    st.subheader("Knowledge Graph")
    st.metric("Nodes", ace_metrics["knowledge_graph_size"])
    st.metric("Relationships", ace_metrics["total_relationships"])
    
    # Recent adaptations
    st.subheader("Recent Adaptations")
    adaptation_history = ace_system.adaptive_pipeline.adaptation_history
    for adaptation in adaptation_history[-5:]:
        st.write(f"**{adaptation['timestamp']}**: {adaptation['adaptation_actions']}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **ACE Not Learning**
   - Check if feedback is being collected
   - Verify API connections
   - Review learning metrics

2. **Poor Performance**
   - Increase user feedback
   - Add expert corrections
   - Check knowledge graph quality

3. **Memory Issues**
   - Monitor knowledge graph size
   - Implement periodic cleanup
   - Optimize node storage

### Debug Commands

```python
# Check ACE system status
def debug_ace_system():
    ace_metrics = ace_system.get_ace_metrics()
    
    print("ACE System Status:")
    print(f"  Knowledge graph size: {ace_metrics['knowledge_graph_size']}")
    print(f"  Total relationships: {ace_metrics['total_relationships']}")
    print(f"  Learning metrics: {ace_metrics['learning_metrics']}")
    
    # Check knowledge graph
    context_engine = ace_system.context_engine
    print(f"  Knowledge nodes: {len(context_engine.knowledge_graph)}")
    print(f"  Relationships: {len(context_engine.relationships)}")
    
    # Check feedback history
    feedback_count = len(ace_system.feedback_collector.feedback_history)
    print(f"  Feedback entries: {feedback_count}")
```

## 🚀 Best Practices

### 1. Start Small
- Begin with basic ACE integration
- Gradually add advanced features
- Monitor performance closely

### 2. Collect Feedback
- Encourage user feedback
- Provide expert corrections
- Track learning progress

### 3. Monitor Performance
- Regular metric reviews
- Knowledge graph analysis
- Adaptation tracking

### 4. Backup and Recovery
- Save ACE state regularly
- Backup knowledge graph
- Test recovery procedures

## 📚 Next Steps

1. **Test Integration**: Run `python test_ace_setup.py`
2. **Run Demo**: Execute `python ace_demo.py`
3. **Start Application**: Launch `streamlit run shifter_rag_app_ace.py`
4. **Monitor Learning**: Track metrics and performance
5. **Iterate and Improve**: Based on user feedback and system performance

---

*Transform your LHCb RAG system into an autonomous, self-improving AI assistant with ACE!* 🧠⚡
