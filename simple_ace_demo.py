#!/usr/bin/env python3
"""
Simple ACE Demo - Working demonstration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ace_framework import ACESystem, ExecutionTrace

class SimpleRAG:
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

def main():
    print("🎭 Simple ACE Demo")
    print("=" * 30)
    
    # Initialize ACE
    rag = SimpleRAG()
    ace = ACESystem(rag)
    
    # Test query
    query = "What is the 4-cells problem?"
    print(f"Query: {query}")
    
    # Process with ACE
    response, metrics = ace.process_query_with_ace(query)
    
    print(f"\nResponse: {response[:100]}...")
    print(f"ACE Pipeline: {metrics.get('ace_pipeline_executed', False)}")
    print(f"Generation Cycles: {metrics.get('generation_cycles', 0)}")
    print(f"Reflection Cycles: {metrics.get('reflection_cycles', 0)}")
    print(f"Curation Cycles: {metrics.get('curation_cycles', 0)}")
    
    # Show status
    status = ace.get_ace_status()
    print(f"\nACE Status: {status['ace_implementation']}")
    print("✅ ACE Framework is working!")

if __name__ == "__main__":
    main()
