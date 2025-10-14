#!/usr/bin/env python3
"""
Demonstration script for LHCb Shifter Assistant with Agentic Context Engineering (ACE)
Shows how the ACE system learns and improves over time.
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_ace_learning():
    """Demonstrate ACE learning capabilities"""
    print("🧠 ACE Learning Demonstration")
    print("=" * 50)
    
    try:
        from ace_framework import ACESystem
        from shifter_rag_app_ace import ShifterRAGSystem
        
        # Create a mock base RAG system for demonstration
        class MockRAGSystem:
            def __init__(self):
                self.search_engine = type('MockSearch', (), {
                    'search': lambda self, query, k=5: [
                        {'content': f'Mock document content for: {query}', 'metadata': {'filename': 'demo_doc.pdf'}}
                    ]
                })()
                self.memory_system = type('MockMemory', (), {})()
            
            def generate_response(self, query, context_docs=None, use_memory=True):
                return f"Base RAG response for: {query}"
        
        # Initialize ACE system
        print("1. Initializing ACE system...")
        mock_rag = MockRAGSystem()
        ace_system = ACESystem(mock_rag)
        print("   ✅ ACE system initialized")
        
        # Demonstrate initial query
        print("\n2. First query (no learning yet)...")
        query1 = "How do I restart the cooling system?"
        response1, metrics1 = ace_system.process_query_with_ace(query1)
        print(f"   Query: {query1}")
        print(f"   Response: {response1}")
        print(f"   ACE Metrics: {metrics1}")
        
        # Add some context nodes to simulate learning
        print("\n3. Adding knowledge to ACE system...")
        ace_system.context_engine.add_context_node(
            content="Cooling system restart procedure: 1. Check pressure readings 2. Close valves 3. Restart pump",
            category="procedures",
            source="expert_manual",
            confidence=0.9
        )
        ace_system.context_engine.add_context_node(
            content="Safety warning: Always check pressure before restarting cooling system",
            category="safety",
            source="safety_manual",
            confidence=0.95
        )
        print("   ✅ Knowledge nodes added to ACE system")
        
        # Demonstrate learning from feedback
        print("\n4. Learning from user feedback...")
        feedback_result = ace_system.collect_feedback(
            query1, response1, 4,  # Good rating
            expert_correction="Also check temperature sensors before restart",
            improvement_suggestions=["Include temperature checks in procedure"]
        )
        print(f"   Feedback collected: {feedback_result['feedback_id']}")
        print(f"   Evolution actions: {feedback_result['evolution_result']['evolution_actions']}")
        
        # Show improved response
        print("\n5. Second query (with learning applied)...")
        query2 = "What safety checks should I do before restarting cooling?"
        response2, metrics2 = ace_system.process_query_with_ace(query2)
        print(f"   Query: {query2}")
        print(f"   Response: {response2}")
        print(f"   ACE Metrics: {metrics2}")
        
        # Demonstrate negative feedback learning
        print("\n6. Learning from negative feedback...")
        feedback_result2 = ace_system.collect_feedback(
            query2, response2, 2,  # Poor rating
            expert_correction="Include pressure gauge readings and valve positions",
            improvement_suggestions=["Be more specific about safety checks"]
        )
        print(f"   Negative feedback processed")
        print(f"   Adaptation actions: {feedback_result2['adaptation_actions']}")
        
        # Show final ACE metrics
        print("\n7. Final ACE system metrics...")
        final_metrics = ace_system.get_ace_metrics()
        print(f"   Knowledge graph size: {final_metrics['knowledge_graph_size']}")
        print(f"   Total relationships: {final_metrics['total_relationships']}")
        print(f"   Learning metrics: {final_metrics['learning_metrics']}")
        
        print("\n🎉 ACE learning demonstration completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure ace_framework.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return False

def demo_knowledge_graph_evolution():
    """Demonstrate knowledge graph evolution"""
    print("\n🧠 Knowledge Graph Evolution Demo")
    print("=" * 50)
    
    try:
        from ace_framework import ContextEvolutionEngine
        
        # Initialize context evolution engine
        engine = ContextEvolutionEngine()
        print("1. Initializing context evolution engine...")
        
        # Add initial knowledge
        print("2. Adding initial knowledge nodes...")
        node1 = engine.add_context_node(
            content="PLUME system monitoring procedures",
            category="procedures",
            source="manual",
            confidence=0.8
        )
        node2 = engine.add_context_node(
            content="PLUME alarm handling steps",
            category="troubleshooting",
            source="manual",
            confidence=0.7
        )
        print(f"   Added nodes: {node1}, {node2}")
        
        # Simulate feedback and evolution
        print("3. Simulating feedback and evolution...")
        from ace_framework import FeedbackEntry
        
        feedback = FeedbackEntry(
            id="demo_feedback",
            query="How do I handle PLUME alarms?",
            response="Check PLUME system status and follow alarm procedures",
            user_rating=4,
            expert_correction="Also verify detector connections",
            timestamp=datetime.now()
        )
        
        evolution_result = engine.evolve_context(feedback)
        print(f"   Evolution result: {evolution_result}")
        
        # Show knowledge graph state
        print("4. Knowledge graph state...")
        print(f"   Total nodes: {len(engine.knowledge_graph)}")
        print(f"   Total relationships: {sum(len(rels) for rels in engine.relationships.values())}")
        
        for node_id, node in engine.knowledge_graph.items():
            print(f"   Node {node_id}: {node.category} (confidence: {node.confidence:.2f})")
        
        print("\n✅ Knowledge graph evolution demonstration completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in knowledge graph demo: {e}")
        return False

def demo_adaptive_rag():
    """Demonstrate adaptive RAG pipeline"""
    print("\n🔄 Adaptive RAG Pipeline Demo")
    print("=" * 50)
    
    try:
        from ace_framework import ACESystem, AdaptiveRAGPipeline, ContextEvolutionEngine
        
        # Create mock base RAG
        class MockRAGSystem:
            def __init__(self):
                self.search_engine = type('MockSearch', (), {
                    'search': lambda self, query, k=5: [
                        {'content': f'Base context for: {query}', 'metadata': {'filename': 'base_doc.pdf'}}
                    ]
                })()
            
            def generate_response(self, query, context_docs=None, use_memory=True):
                return f"Base RAG response for: {query}"
        
        # Initialize adaptive pipeline
        print("1. Initializing adaptive RAG pipeline...")
        base_rag = MockRAGSystem()
        context_engine = ContextEvolutionEngine()
        adaptive_pipeline = AdaptiveRAGPipeline(base_rag, context_engine)
        
        # Add some learned context
        print("2. Adding learned context...")
        context_engine.add_context_node(
            content="Advanced troubleshooting for detector systems",
            category="troubleshooting",
            source="learned",
            confidence=0.9
        )
        
        # Generate adaptive response
        print("3. Generating adaptive response...")
        query = "Detector system not responding"
        response, metrics = adaptive_pipeline.generate_adaptive_response(query)
        print(f"   Query: {query}")
        print(f"   Response: {response}")
        print(f"   Metrics: {metrics}")
        
        # Simulate learning from feedback
        print("4. Learning from feedback...")
        from ace_framework import FeedbackEntry
        
        feedback = FeedbackEntry(
            id="adaptive_feedback",
            query=query,
            response=response,
            user_rating=5,
            timestamp=datetime.now()
        )
        
        evolution_result, adaptation_actions = adaptive_pipeline.learn_from_feedback(feedback)
        print(f"   Evolution result: {evolution_result}")
        print(f"   Adaptation actions: {adaptation_actions}")
        
        print("\n✅ Adaptive RAG demonstration completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in adaptive RAG demo: {e}")
        return False

def main():
    """Run all ACE demonstrations"""
    print("🧠 LHCb Shifter Assistant with Agentic Context Engineering (ACE)")
    print("Demonstration Script")
    print("=" * 60)
    
    demos = [
        ("ACE Learning", demo_ace_learning),
        ("Knowledge Graph Evolution", demo_knowledge_graph_evolution),
        ("Adaptive RAG Pipeline", demo_adaptive_rag)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            result = demo_func()
            results.append((demo_name, result))
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 DEMONSTRATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for demo_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{demo_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} demonstrations passed")
    
    if passed == total:
        print("\n🎉 All ACE demonstrations completed successfully!")
        print("\n🚀 To run the full ACE system:")
        print("   streamlit run shifter_rag_app_ace.py")
    else:
        print("\n⚠️ Some demonstrations failed. Check the errors above.")
        print("\n🔧 To troubleshoot:")
        print("   1. Run: python test_ace_setup.py")
        print("   2. Check that ace_framework.py is present")
        print("   3. Verify all dependencies are installed")

if __name__ == "__main__":
    main()
