#!/usr/bin/env python3
"""
Test ACE integration in the Streamlit app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ace_framework import ACESystem

class TestRAG:
    def __init__(self):
        self.search_results = [
            {"content": "4-cells problem: 4 consecutive vertical cells showing noise patterns"},
            {"content": "ECAL detector monitoring requires checking cell patterns"},
            {"content": "Noisy cells indicate potential calibration issues"}
        ]
        self.search_engine = self
    
    def search(self, query, k=3):
        return self.search_results[:k]
    
    def generate_response(self, query, contexts, use_memory=True):
        return f"Based on context: {', '.join(contexts[:2])}, here's the answer to: {query}"

def test_ace_learning():
    """Test if ACE is actually learning from feedback"""
    print("🧪 Testing ACE Learning Integration")
    print("=" * 50)
    
    # Initialize ACE system
    rag = TestRAG()
    ace = ACESystem(rag)
    
    # Test query
    query = "What is the 4-cells problem?"
    print(f"📝 Query: {query}")
    
    # Process with ACE
    response, metrics = ace.process_query_with_ace(query)
    print(f"📊 Response: {response[:100]}...")
    print(f"🎯 ACE Pipeline Executed: {metrics.get('ace_pipeline_executed', False)}")
    print(f"🔄 Generation Cycles: {metrics.get('generation_cycles', 0)}")
    print(f"🔍 Reflection Cycles: {metrics.get('reflection_cycles', 0)}")
    print(f"📚 Curation Cycles: {metrics.get('curation_cycles', 0)}")
    
    # Provide feedback
    print(f"\n🎓 Providing Expert Feedback...")
    feedback_result = ace.collect_feedback(
        query=query,
        response=response,
        user_rating=3,  # Neutral rating
        expert_correction="When 4 consecutive vertical cells turn noisy, restart the affected system (ECAL or HCAL), or only the specific board if you can find it without losing too much time of data taking with this problem. Shifter should report to ProblemDB (https://lbproblems.cern.ch/problemdb/) - please include the link. Also help the user find the correct or specific board if possible."
    )
    
    print(f"✅ Feedback collected: {feedback_result['feedback_id']}")
    print(f"🧠 Expert knowledge stored: {feedback_result['expert_knowledge_stored']}")
    print(f"📈 Knowledge nodes added: {feedback_result['knowledge_nodes_added']}")
    
    # Test same query again
    print(f"\n🔄 Testing same query again...")
    response2, metrics2 = ace.process_query_with_ace(query)
    print(f"📊 Response 2: {response2[:100]}...")
    print(f"🎯 ACE Pipeline Executed: {metrics2.get('ace_pipeline_executed', False)}")
    print(f"🔄 Generation Cycles: {metrics2.get('generation_cycles', 0)}")
    
    # Check if expert knowledge was retrieved
    expert_knowledge = ace._retrieve_expert_knowledge(query)
    print(f"\n🔍 Expert Knowledge Retrieved: {len(expert_knowledge)} chars")
    if expert_knowledge:
        print(f"📝 Expert Knowledge: {expert_knowledge[:200]}...")
    else:
        print("❌ No expert knowledge retrieved!")
    
    # Get ACE status
    status = ace.get_ace_status()
    print(f"\n🎯 ACE Status: {status['ace_implementation']}")
    print(f"📊 Knowledge Graph Size: {ace.get_ace_metrics()['knowledge_graph_size']}")
    
    return expert_knowledge is not None

if __name__ == "__main__":
    success = test_ace_learning()
    if success:
        print("\n✅ ACE Learning is working!")
    else:
        print("\n❌ ACE Learning is NOT working properly!")
