#!/usr/bin/env python3
"""
Test script for ACE (Agentic Context Engineering) functionality
This script tests all major ACE components to ensure they're working properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ace_framework import ACESystem, ContextNode
from shifter_rag_app_ace_vision import VisionRAGSystem
from datetime import datetime
import json

def test_ace_initialization():
    """Test 1: ACE System Initialization"""
    print("🧪 Test 1: ACE System Initialization")
    try:
        # Initialize base RAG system
        base_rag = VisionRAGSystem()
        print("✅ Base RAG system initialized")
        
        # Initialize ACE system
        ace_system = ACESystem(base_rag)
        print("✅ ACE system initialized")
        
        # Test ACE components
        assert hasattr(ace_system, 'context_engine'), "Context engine missing"
        assert hasattr(ace_system, 'adaptive_pipeline'), "Adaptive pipeline missing"
        assert hasattr(ace_system, 'feedback_collector'), "Feedback collector missing"
        print("✅ All ACE components present")
        
        return ace_system
    except Exception as e:
        print(f"❌ ACE initialization failed: {e}")
        return None

def test_knowledge_graph_operations(ace_system):
    """Test 2: Knowledge Graph Operations"""
    print("\n🧪 Test 2: Knowledge Graph Operations")
    try:
        # Test adding context nodes
        test_content = "Test operational procedure for LHCb equipment"
        ace_system.context_engine.add_context_node(
            content=test_content,
            category="test_procedure",
            source="test_source",
            confidence=0.8
        )
        print("✅ Context node added successfully")
        
        # Test knowledge graph size
        initial_size = len(ace_system.context_engine.knowledge_graph)
        print(f"✅ Knowledge graph size: {initial_size}")
        
        # Test finding relevant nodes
        relevant_nodes = ace_system.context_engine._find_relevant_nodes("LHCb equipment")
        print(f"✅ Found {len(relevant_nodes)} relevant nodes")
        
        return True
    except Exception as e:
        print(f"❌ Knowledge graph operations failed: {e}")
        return False

def test_vision_concept_extraction(ace_system):
    """Test 3: Vision Concept Extraction"""
    print("\n🧪 Test 3: Vision Concept Extraction")
    try:
        # Test image data
        test_image_data = {
            'filename': 'test_4cells.png',
            'dimensions': (1024, 768),
            'size': 50000,
            'base64': 'test_base64_data'
        }
        
        # Test query
        test_query = "What should I do about this 4-cells problem?"
        
        # Test concept extraction
        concepts = ace_system._extract_visual_concepts(test_query, test_image_data)
        print(f"✅ Extracted {len(concepts)} visual concepts")
        
        # Check for 4-cells problem detection
        four_cells_concepts = [c for c in concepts if c['type'] == '4cells_problem']
        if four_cells_concepts:
            print("✅ 4-cells problem pattern detected")
        else:
            print("⚠️ 4-cells problem pattern not detected")
        
        # Check for ECAL/HCAL detection
        ecal_concepts = [c for c in concepts if c['type'] == 'ecal_hcal_analysis']
        if ecal_concepts:
            print("✅ ECAL/HCAL analysis pattern detected")
        
        return True
    except Exception as e:
        print(f"❌ Vision concept extraction failed: {e}")
        return False

def test_ace_learning(ace_system):
    """Test 4: ACE Learning Capabilities"""
    print("\n🧪 Test 4: ACE Learning Capabilities")
    try:
        # Test query processing with ACE
        test_query = "How do I handle a 4-cells problem in ECAL?"
        response, ace_metrics = ace_system.process_query_with_ace(test_query, use_memory=True)
        
        print(f"✅ ACE response generated: {len(response)} characters")
        print(f"✅ ACE metrics: {ace_metrics}")
        
        # Test vision query processing
        test_image_data = {
            'filename': '4cells_problem.png',
            'dimensions': (1024, 768),
            'size': 50000,
            'base64': 'test_base64_data'
        }
        
        vision_response, vision_metrics = ace_system.process_vision_query_with_ace(
            test_query, test_image_data, use_memory=True
        )
        
        print(f"✅ Vision response generated: {len(vision_response)} characters")
        print(f"✅ Vision metrics: {vision_metrics}")
        
        return True
    except Exception as e:
        print(f"❌ ACE learning failed: {e}")
        return False

def test_ace_metrics(ace_system):
    """Test 5: ACE Metrics and Evaluation"""
    print("\n🧪 Test 5: ACE Metrics and Evaluation")
    try:
        # Get ACE metrics
        metrics = ace_system.get_ace_metrics()
        print(f"✅ ACE metrics retrieved: {len(metrics)} metrics")
        
        # Display key metrics
        print(f"   📊 Knowledge graph size: {metrics.get('knowledge_graph_size', 0)}")
        print(f"   📊 Total relationships: {metrics.get('total_relationships', 0)}")
        print(f"   📊 Adaptation count: {metrics.get('adaptation_count', 0)}")
        print(f"   📊 Feedback count: {metrics.get('feedback_count', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ ACE metrics failed: {e}")
        return False

def test_feedback_learning(ace_system):
    """Test 6: Feedback Learning System"""
    print("\n🧪 Test 6: Feedback Learning System")
    try:
        # Test feedback collection
        feedback_result = ace_system.collect_feedback(
            query="Test query about 4-cells problem",
            response="Test response about restarting ECAL system",
            user_rating=5,
            expert_correction="This is the correct procedure for 4-cells problems",
            improvement_suggestions=["Add more specific timing information"]
        )
        
        print(f"✅ Feedback collected: {feedback_result['feedback_id']}")
        print(f"✅ Evolution result: {feedback_result['evolution_result']}")
        print(f"✅ Adaptation actions: {feedback_result['adaptation_actions']}")
        
        return True
    except Exception as e:
        print(f"❌ Feedback learning failed: {e}")
        return False

def test_ace_persistence(ace_system):
    """Test 7: ACE State Persistence"""
    print("\n🧪 Test 7: ACE State Persistence")
    try:
        # Test saving ACE state
        test_path = "test_ace_state"
        ace_system.save_ace_state(test_path)
        print("✅ ACE state saved successfully")
        
        # Test loading ACE state
        ace_system.load_ace_state(test_path)
        print("✅ ACE state loaded successfully")
        
        # Clean up test files
        import os
        for ext in ['.pkl', '.json']:
            if os.path.exists(f"{test_path}{ext}"):
                os.remove(f"{test_path}{ext}")
        print("✅ Test files cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ ACE persistence failed: {e}")
        return False

def main():
    """Run all ACE tests"""
    print("🧠 ACE (Agentic Context Engineering) Functionality Test")
    print("=" * 60)
    
    # Test 1: Initialize ACE system
    ace_system = test_ace_initialization()
    if not ace_system:
        print("\n❌ ACE system initialization failed. Cannot continue tests.")
        return
    
    # Run all tests
    tests = [
        test_knowledge_graph_operations,
        test_vision_concept_extraction,
        test_ace_learning,
        test_ace_metrics,
        test_feedback_learning,
        test_ace_persistence
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        if test_func(ace_system):
            passed_tests += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🧪 ACE Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All ACE tests passed! The system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    print("\n🚀 To test with real data, run:")
    print("   streamlit run shifter_rag_app_ace_vision.py")

if __name__ == "__main__":
    main()
