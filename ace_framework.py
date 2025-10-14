"""
Agentic Context Engineering (ACE) Framework for LHCb RAG System

This module implements the core ACE framework that enables autonomous self-improvement
of the RAG system through iterative context evolution and adaptive learning.
"""

import streamlit as st
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re
import hashlib
from pathlib import Path

@dataclass
class ContextNode:
    """Represents a node in the evolving knowledge graph"""
    id: str
    content: str
    category: str
    confidence: float
    source: str
    created_at: datetime
    updated_at: datetime
    relationships: List[str]
    feedback_score: float = 0.0
    usage_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class FeedbackEntry:
    """Represents user feedback for continuous learning"""
    id: str
    query: str
    response: str
    user_rating: int  # 1-5 scale
    expert_correction: Optional[str] = None
    timestamp: datetime = None
    context_used: List[str] = None
    improvement_suggestions: List[str] = None

@dataclass
class LearningMetrics:
    """Tracks learning and improvement metrics"""
    total_interactions: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    context_evolution_count: int = 0
    knowledge_growth_rate: float = 0.0
    accuracy_improvement: float = 0.0
    response_quality_score: float = 0.0

class ContextEvolutionEngine:
    """Core engine for autonomous context evolution"""
    
    def __init__(self):
        self.knowledge_graph = {}  # id -> ContextNode
        self.relationships = defaultdict(list)  # id -> [related_ids]
        self.feedback_history = []
        self.learning_metrics = LearningMetrics()
        self.evolution_threshold = 0.7  # Threshold for triggering evolution
        self.decay_factor = 0.95  # Knowledge decay over time
        
    def add_context_node(self, content: str, category: str, source: str, 
                        confidence: float = 0.8) -> str:
        """Add a new context node to the knowledge graph"""
        node_id = self._generate_node_id(content, category)
        
        # Check if similar node exists
        existing_node = self._find_similar_node(content, category)
        if existing_node:
            # Update existing node
            self._update_node(existing_node, content, confidence)
            return existing_node
        
        # Create new node
        node = ContextNode(
            id=node_id,
            content=content,
            category=category,
            confidence=confidence,
            source=source,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            relationships=[]
        )
        
        self.knowledge_graph[node_id] = node
        self._establish_relationships(node)
        return node_id
    
    def evolve_context(self, feedback: FeedbackEntry) -> Dict[str, Any]:
        """Evolve context based on feedback and learning"""
        evolution_actions = []
        
        # Analyze feedback for learning opportunities
        if feedback.user_rating >= 4:
            # Positive feedback - strengthen relevant contexts
            self._strengthen_contexts(feedback.context_used or [])
            evolution_actions.append("strengthened_positive_contexts")
        elif feedback.user_rating <= 2:
            # Negative feedback - identify and improve weak contexts
            weak_contexts = self._identify_weak_contexts(feedback)
            self._improve_contexts(weak_contexts, feedback.expert_correction)
            evolution_actions.append("improved_weak_contexts")
        
        # Update learning metrics
        self._update_learning_metrics(feedback)
        
        # Trigger knowledge graph evolution if threshold met
        if self._should_evolve():
            evolution_result = self._trigger_evolution()
            evolution_actions.extend(evolution_result)
        
        return {
            "evolution_actions": evolution_actions,
            "metrics_updated": True,
            "knowledge_growth": self.learning_metrics.knowledge_growth_rate
        }
    
    def get_evolved_context(self, query: str, base_contexts: List[str]) -> List[str]:
        """Get evolved context for a query using learned relationships"""
        # Find relevant nodes
        relevant_nodes = self._find_relevant_nodes(query)
        
        # Apply relationship learning
        enhanced_contexts = self._apply_relationship_learning(base_contexts, relevant_nodes)
        
        # Update usage statistics
        for node_id in relevant_nodes:
            if node_id in self.knowledge_graph:
                self.knowledge_graph[node_id].usage_count += 1
                self.knowledge_graph[node_id].last_accessed = datetime.now()
        
        return enhanced_contexts
    
    def _generate_node_id(self, content: str, category: str) -> str:
        """Generate unique node ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{category}_{content_hash}"
    
    def _find_similar_node(self, content: str, category: str) -> Optional[str]:
        """Find existing similar node"""
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        for node_id, node in self.knowledge_graph.items():
            if node.category == category:
                node_words = set(re.findall(r'\b\w+\b', node.content.lower()))
                similarity = len(content_words.intersection(node_words)) / len(content_words.union(node_words))
                if similarity > 0.8:  # High similarity threshold
                    return node_id
        return None
    
    def _update_node(self, node_id: str, content: str, confidence: float):
        """Update existing node with new information"""
        if node_id in self.knowledge_graph:
            node = self.knowledge_graph[node_id]
            # Merge content intelligently
            node.content = self._merge_content(node.content, content)
            node.confidence = max(node.confidence, confidence)
            node.updated_at = datetime.now()
    
    def _merge_content(self, existing: str, new: str) -> str:
        """Intelligently merge content from similar contexts"""
        # Simple merge strategy - could be enhanced with NLP
        existing_sentences = existing.split('. ')
        new_sentences = new.split('. ')
        
        # Add new unique sentences
        for sentence in new_sentences:
            if sentence.strip() and sentence not in existing_sentences:
                existing_sentences.append(sentence)
        
        return '. '.join(existing_sentences)
    
    def _establish_relationships(self, node: ContextNode):
        """Establish relationships between nodes"""
        node_words = set(re.findall(r'\b\w+\b', node.content.lower()))
        
        for other_id, other_node in self.knowledge_graph.items():
            if other_id != node.id:
                other_words = set(re.findall(r'\b\w+\b', other_node.content.lower()))
                overlap = len(node_words.intersection(other_words))
                
                if overlap > 3:  # Minimum word overlap for relationship
                    self.relationships[node.id].append(other_id)
                    self.relationships[other_id].append(node.id)
    
    def _find_relevant_nodes(self, query: str) -> List[str]:
        """Find relevant nodes for a query"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        relevant_nodes = []
        
        for node_id, node in self.knowledge_graph.items():
            node_words = set(re.findall(r'\b\w+\b', node.content.lower()))
            overlap = len(query_words.intersection(node_words))
            
            if overlap > 0:
                relevance_score = overlap / len(query_words)
                relevant_nodes.append((node_id, relevance_score))
        
        # Sort by relevance and return top nodes
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in relevant_nodes[:5]]
    
    def _apply_relationship_learning(self, base_contexts: List[str], relevant_nodes: List[str]) -> List[str]:
        """Apply learned relationships to enhance context"""
        enhanced_contexts = base_contexts.copy()
        
        for node_id in relevant_nodes:
            if node_id in self.knowledge_graph:
                node = self.knowledge_graph[node_id]
                
                # Add related nodes' content
                for related_id in self.relationships[node_id]:
                    if related_id in self.knowledge_graph:
                        related_node = self.knowledge_graph[related_id]
                        if related_node.content not in enhanced_contexts:
                            enhanced_contexts.append(related_node.content)
        
        return enhanced_contexts
    
    def _strengthen_contexts(self, context_ids: List[str]):
        """Strengthen contexts that received positive feedback"""
        for context_id in context_ids:
            if context_id in self.knowledge_graph:
                node = self.knowledge_graph[context_id]
                node.confidence = min(1.0, node.confidence + 0.1)
                node.feedback_score += 0.1
    
    def _identify_weak_contexts(self, feedback: FeedbackEntry) -> List[str]:
        """Identify contexts that need improvement"""
        weak_contexts = []
        
        # Find contexts with low confidence or negative feedback
        for node_id, node in self.knowledge_graph.items():
            if (node.confidence < 0.6 or 
                node.feedback_score < 0 or 
                (node.last_accessed and 
                 (datetime.now() - node.last_accessed).days > 30)):
                weak_contexts.append(node_id)
        
        return weak_contexts
    
    def _improve_contexts(self, weak_contexts: List[str], expert_correction: Optional[str]):
        """Improve weak contexts using expert feedback"""
        for context_id in weak_contexts:
            if context_id in self.knowledge_graph:
                node = self.knowledge_graph[context_id]
                
                if expert_correction:
                    # Update content with expert correction
                    node.content = expert_correction
                    node.confidence = 0.9  # High confidence for expert corrections
                
                # Mark for review
                node.updated_at = datetime.now()
    
    def _update_learning_metrics(self, feedback: FeedbackEntry):
        """Update learning metrics based on feedback"""
        self.learning_metrics.total_interactions += 1
        
        if feedback.user_rating >= 4:
            self.learning_metrics.positive_feedback += 1
        elif feedback.user_rating <= 2:
            self.learning_metrics.negative_feedback += 1
        
        # Calculate accuracy improvement
        if self.learning_metrics.total_interactions > 0:
            accuracy = self.learning_metrics.positive_feedback / self.learning_metrics.total_interactions
            self.learning_metrics.accuracy_improvement = accuracy
        
        # Calculate response quality score
        self.learning_metrics.response_quality_score = (
            self.learning_metrics.positive_feedback * 2 + 
            self.learning_metrics.total_interactions
        ) / (self.learning_metrics.total_interactions * 2)
    
    def _should_evolve(self) -> bool:
        """Determine if knowledge graph should evolve"""
        if self.learning_metrics.total_interactions < 10:
            return False
        
        # Evolve if accuracy is below threshold
        accuracy = self.learning_metrics.positive_feedback / max(1, self.learning_metrics.total_interactions)
        return accuracy < self.evolution_threshold
    
    def _trigger_evolution(self) -> List[str]:
        """Trigger knowledge graph evolution"""
        evolution_actions = []
        
        # Remove outdated or low-quality nodes
        nodes_to_remove = []
        for node_id, node in self.knowledge_graph.items():
            if (node.confidence < 0.3 or 
                (node.last_accessed and 
                 (datetime.now() - node.last_accessed).days > 90)):
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del self.knowledge_graph[node_id]
            evolution_actions.append(f"removed_outdated_node_{node_id}")
        
        # Strengthen high-quality nodes
        for node_id, node in self.knowledge_graph.items():
            if node.confidence > 0.8 and node.feedback_score > 0:
                node.confidence = min(1.0, node.confidence + 0.05)
                evolution_actions.append(f"strengthened_high_quality_node_{node_id}")
        
        self.learning_metrics.context_evolution_count += 1
        return evolution_actions
    
    def save_knowledge_graph(self, path: str):
        """Save knowledge graph to disk"""
        data = {
            'knowledge_graph': {k: asdict(v) for k, v in self.knowledge_graph.items()},
            'relationships': dict(self.relationships),
            'learning_metrics': asdict(self.learning_metrics)
        }
        
        with open(f"{path}_ace.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load_knowledge_graph(self, path: str):
        """Load knowledge graph from disk"""
        try:
            if Path(f"{path}_ace.pkl").exists():
                with open(f"{path}_ace.pkl", 'rb') as f:
                    data = pickle.load(f)
                
                # Reconstruct knowledge graph
                self.knowledge_graph = {}
                for k, v in data['knowledge_graph'].items():
                    # Convert datetime strings back to datetime objects
                    v['created_at'] = datetime.fromisoformat(v['created_at'])
                    v['updated_at'] = datetime.fromisoformat(v['updated_at'])
                    if v['last_accessed']:
                        v['last_accessed'] = datetime.fromisoformat(v['last_accessed'])
                    self.knowledge_graph[k] = ContextNode(**v)
                
                self.relationships = defaultdict(list, data['relationships'])
                self.learning_metrics = LearningMetrics(**data['learning_metrics'])
                
                return True
        except Exception as e:
            st.error(f"Error loading ACE knowledge graph: {str(e)}")
            return False

class AdaptiveRAGPipeline:
    """Adaptive RAG pipeline that learns and improves over time"""
    
    def __init__(self, base_rag_system, context_evolution_engine: ContextEvolutionEngine):
        self.base_rag = base_rag_system
        self.context_engine = context_evolution_engine
        self.adaptation_history = []
        self.performance_tracker = {}
        
    def generate_adaptive_response(self, query: str, use_memory: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Generate response using adaptive RAG pipeline"""
        # Get base context from original RAG
        base_contexts = self._get_base_contexts(query)
        
        # Apply ACE context evolution
        evolved_contexts = self.context_engine.get_evolved_context(query, base_contexts)
        
        # Generate response with evolved context
        response = self._generate_response_with_context(query, evolved_contexts, use_memory)
        
        # Track performance
        self._track_performance(query, response, evolved_contexts)
        
        return response, {
            "base_contexts_count": len(base_contexts),
            "evolved_contexts_count": len(evolved_contexts),
            "context_enhancement_ratio": len(evolved_contexts) / max(1, len(base_contexts)),
            "ace_applied": len(evolved_contexts) > len(base_contexts)
        }
    
    def learn_from_feedback(self, feedback: FeedbackEntry):
        """Learn from user feedback to improve future responses"""
        # Evolve context based on feedback
        evolution_result = self.context_engine.evolve_context(feedback)
        
        # Adapt RAG pipeline based on learning
        adaptation_actions = self._adapt_pipeline(feedback, evolution_result)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now(),
            "feedback_id": feedback.id,
            "evolution_result": evolution_result,
            "adaptation_actions": adaptation_actions
        })
        
        return evolution_result, adaptation_actions
    
    def _get_base_contexts(self, query: str) -> List[str]:
        """Get base contexts from original RAG system"""
        search_results = self.base_rag.search_engine.search(query, k=3)
        return [result['content'] for result in search_results]
    
    def _generate_response_with_context(self, query: str, contexts: List[str], use_memory: bool) -> str:
        """Generate response using evolved contexts"""
        # Use the base RAG system but with evolved contexts
        return self.base_rag.generate_response(query, contexts, use_memory)
    
    def _track_performance(self, query: str, response: str, contexts: List[str]):
        """Track performance metrics for continuous improvement"""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        self.performance_tracker[query_hash] = {
            "timestamp": datetime.now(),
            "context_count": len(contexts),
            "response_length": len(response),
            "query_complexity": len(query.split())
        }
    
    def _adapt_pipeline(self, feedback: FeedbackEntry, evolution_result: Dict[str, Any]) -> List[str]:
        """Adapt pipeline based on feedback and evolution"""
        adaptation_actions = []
        
        # Adjust search parameters based on feedback quality
        if feedback.user_rating >= 4:
            # Positive feedback - maintain current approach
            adaptation_actions.append("maintained_positive_approach")
        elif feedback.user_rating <= 2:
            # Negative feedback - adjust search strategy
            adaptation_actions.append("adjusted_search_strategy")
            
            # Increase context diversity for similar queries
            if "improved_weak_contexts" in evolution_result.get("evolution_actions", []):
                adaptation_actions.append("increased_context_diversity")
        
        return adaptation_actions

class ACESystem:
    """Main ACE system that orchestrates all components"""
    
    def __init__(self, base_rag_system):
        # Use enhanced components for self-learning
        self.context_engine = EnhancedContextEvolutionEngine()
        self.adaptive_pipeline = AdaptiveRAGPipeline(base_rag_system, self.context_engine)
        self.feedback_collector = EnhancedFeedbackCollector()
        self.evaluation_system = ACEEvaluationSystem()
        
        # Add new self-learning components
        self.document_refresh_manager = DocumentRefreshManager()
        self.multimodal_learning_engine = MultiModalLearningEngine()
        
        # Load existing knowledge graph if available
        self.load_knowledge_graph()
        
    def process_query_with_ace(self, query: str, use_memory: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Process query using ACE framework"""
        # CRITICAL: Retrieve and apply stored expert knowledge
        expert_knowledge = self._retrieve_expert_knowledge(query)
        
        # Enhance query with expert knowledge if available
        enhanced_query = query
        if expert_knowledge:
            enhanced_query = f"{query}\n\nExpert Knowledge: {expert_knowledge}"
        
        # Generate adaptive response using enhanced query
        response, ace_metrics = self.adaptive_pipeline.generate_adaptive_response(enhanced_query, use_memory)
        
        # Add ACE-specific context nodes if relevant
        self._extract_and_store_context(query, response)
        
        # Update metrics to show expert knowledge was used
        if expert_knowledge:
            ace_metrics["expert_knowledge_used"] = True
            ace_metrics["context_enhanced"] = True
            ace_metrics["ace_applied"] = True
        
        return response, ace_metrics
    
    def collect_feedback(self, query: str, response: str, user_rating: int, 
                        expert_correction: str = None, improvement_suggestions: List[str] = None):
        """Collect and process user feedback"""
        feedback = self.feedback_collector.create_feedback(
            query, response, user_rating, expert_correction, improvement_suggestions
        )
        
        # Learn from feedback
        evolution_result, adaptation_actions = self.adaptive_pipeline.learn_from_feedback(feedback)
        
        # CRITICAL: Actually store expert knowledge in knowledge graph
        if expert_correction:
            # Add expert correction as high-confidence knowledge node
            expert_node_id = f"expert_{hashlib.md5(expert_correction.encode()).hexdigest()[:8]}"
            expert_node = ContextNode(
                id=expert_node_id,
                content=expert_correction,
                category="expert_knowledge",
                confidence=0.95,  # High confidence for expert knowledge
                source="expert_feedback",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                relationships=[],
                feedback_score=user_rating / 5.0,  # Convert rating to feedback score
                usage_count=0,
                last_accessed=None
            )
            self.context_engine.knowledge_graph[expert_node_id] = expert_node
            
            # Create relationships to relevant existing nodes
            query_nodes = self.context_engine._find_relevant_nodes(query)
            for node_id in query_nodes:
                if expert_node_id not in self.context_engine.relationships[node_id]:
                    self.context_engine.relationships[node_id].append(expert_node_id)
                if node_id not in self.context_engine.relationships[expert_node_id]:
                    self.context_engine.relationships[expert_node_id].append(node_id)
            
            # Update learning metrics
            self.context_engine.learning_metrics.knowledge_growth_rate += 0.1
            self.context_engine.learning_metrics.total_interactions += 1
            if user_rating >= 4:
                self.context_engine.learning_metrics.positive_feedback += 1
            else:
                self.context_engine.learning_metrics.negative_feedback += 1
            
            # Save knowledge graph to disk
            self.save_knowledge_graph()
        
        # Store improvement suggestions
        if improvement_suggestions:
            for suggestion in improvement_suggestions:
                suggestion_node_id = f"suggestion_{hashlib.md5(suggestion.encode()).hexdigest()[:8]}"
                suggestion_node = ContextNode(
                    id=suggestion_node_id,
                    content=suggestion,
                    category="improvement_suggestion",
                    confidence=0.8,
                    source="user_feedback",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    relationships=[],
                    feedback_score=user_rating / 5.0,
                    usage_count=0,
                    last_accessed=None
                )
                self.context_engine.knowledge_graph[suggestion_node_id] = suggestion_node
        
        return {
            "feedback_id": feedback.id,
            "evolution_result": evolution_result,
            "adaptation_actions": adaptation_actions,
            "expert_knowledge_stored": expert_correction is not None,
            "knowledge_nodes_added": 1 if expert_correction else 0
        }
    
    def get_ace_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ACE metrics"""
        return {
            "knowledge_graph_size": len(self.context_engine.knowledge_graph),
            "total_relationships": sum(len(rels) for rels in self.context_engine.relationships.values()),
            "learning_metrics": asdict(self.context_engine.learning_metrics),
            "adaptation_count": len(self.adaptive_pipeline.adaptation_history),
            "performance_tracker_size": len(self.adaptive_pipeline.performance_tracker)
        }
    
    def _extract_and_store_context(self, query: str, response: str):
        """Extract and store relevant context from query-response pairs"""
        # Extract key concepts from query and response
        query_concepts = self._extract_concepts(query)
        response_concepts = self._extract_concepts(response)
        
        # Store as context nodes
        for concept in query_concepts + response_concepts:
            if len(concept) > 10:  # Only store substantial concepts
                self.context_engine.add_context_node(
                    content=concept,
                    category="extracted_concept",
                    source="ace_extraction",
                    confidence=0.7
                )
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction - could be enhanced with NLP
        sentences = text.split('. ')
        concepts = []
        
        for sentence in sentences:
            if len(sentence.split()) > 5:  # Substantial sentences
                concepts.append(sentence.strip())
        
        return concepts
    
    def process_vision_query_with_ace(self, query: str, image_data: Dict, use_memory: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Process vision query with ACE learning capabilities"""
        try:
            # CRITICAL: Retrieve and apply stored expert knowledge
            expert_knowledge = self._retrieve_expert_knowledge(query)
            
            # Enhance query with expert knowledge if available
            enhanced_query = query
            if expert_knowledge:
                enhanced_query = f"{query}\n\nExpert Knowledge: {expert_knowledge}"
            
            # Generate vision response using enhanced query
            vision_response = self.adaptive_pipeline.base_rag.generate_vision_response(
                enhanced_query, image_data, use_memory=use_memory
            )
            
            # Apply ACE learning to vision context
            ace_metrics = {
                "ace_applied": False,
                "knowledge_nodes_added": 0,
                "relationships_created": 0,
                "context_enhanced": False,
                "expert_knowledge_used": len(expert_knowledge) > 0 if expert_knowledge else False
            }
            
            # Extract visual concepts for knowledge graph
            visual_concepts = self._extract_visual_concepts(query, image_data)
            
            if visual_concepts:
                # Add visual knowledge to graph
                for concept in visual_concepts:
                    node_id = f"visual_{concept['type']}_{hashlib.md5(concept['content'].encode()).hexdigest()[:8]}"
                    
                    if node_id not in self.context_engine.knowledge_graph:
                        visual_node = ContextNode(
                            id=node_id,
                            content=concept['content'],
                            category="visual_knowledge",
                            confidence=0.7,
                            source="ace_vision_extraction",
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            relationships=[],
                            feedback_score=0.0,
                            usage_count=0,
                            last_accessed=None
                        )
                        self.context_engine.knowledge_graph[node_id] = visual_node
                        ace_metrics["knowledge_nodes_added"] += 1
                        ace_metrics["ace_applied"] = True
                
                # Create relationships between visual and text knowledge
                text_nodes = self.context_engine._find_relevant_nodes(query)
                for text_node_id in text_nodes:
                    for visual_node_id in [nid for nid in self.context_engine.knowledge_graph.keys() if nid.startswith("visual_")]:
                        if visual_node_id not in self.context_engine.relationships[text_node_id]:
                            self.context_engine.relationships[text_node_id].append(visual_node_id)
                            ace_metrics["relationships_created"] += 1
                            ace_metrics["ace_applied"] = True
            
            # Mark expert knowledge as used
            if expert_knowledge:
                ace_metrics["context_enhanced"] = True
                ace_metrics["ace_applied"] = True
            
            return vision_response, ace_metrics
            
        except Exception as e:
            return f"Error processing vision query with ACE: {str(e)}", {"ace_applied": False}
    
    def _retrieve_expert_knowledge(self, query: str) -> str:
        """Retrieve relevant expert knowledge from the knowledge graph"""
        expert_knowledge = []
        query_lower = query.lower()
        
        # Find expert knowledge nodes
        for node_id, node in self.context_engine.knowledge_graph.items():
            if node.category == "expert_knowledge":
                # Enhanced relevance checking
                node_content_lower = node.content.lower()
                
                # Check for direct keyword matches
                keywords = ["4-cells", "4 cells", "noisy cells", "vertical cells", "ecal", "hcal", "restart", "problem"]
                query_has_keywords = any(keyword in query_lower for keyword in keywords)
                node_has_keywords = any(keyword in node_content_lower for keyword in keywords)
                
                # Check for word overlap (more flexible)
                query_words = set(query_lower.replace("-", " ").replace("_", " ").split())
                node_words = set(node_content_lower.replace("-", " ").replace("_", " ").split())
                overlap = len(query_words.intersection(node_words))
                
                # Check for substring matches
                substring_match = any(word in node_content_lower for word in query_words if len(word) > 3)
                
                # More flexible matching criteria
                if (query_has_keywords and node_has_keywords) or overlap > 0 or substring_match:
                    expert_knowledge.append(node.content)
        
        return " | ".join(expert_knowledge) if expert_knowledge else ""
    
    def analyze_query_patterns(self, query: str, response_quality: float):
        """Track query patterns for document refresh recommendations"""
        self.document_refresh_manager.analyze_query_patterns(query, response_quality)
    
    def get_document_refresh_recommendations(self) -> List[dict]:
        """Get prioritized list of documents that need updating"""
        return self.document_refresh_manager.get_refresh_recommendations()
    
    def detect_contradictions(self, new_node: ContextNode) -> List[str]:
        """Detect contradictions in new knowledge"""
        return self.context_engine.detect_contradictions(new_node)
    
    def apply_confidence_decay(self, days_threshold: int = 30):
        """Apply confidence decay to old knowledge"""
        self.context_engine.apply_confidence_decay(days_threshold)
    
    def learn_from_vision_feedback(self, image_data: dict, analysis: str, expert_correction: str):
        """Learn from vision feedback using multimodal learning"""
        self.multimodal_learning_engine.learn_from_vision_feedback(
            image_data, analysis, expert_correction
        )
    
    def collect_implicit_feedback(self, query: str, response: str, user_behavior: dict):
        """Collect implicit feedback from user behavior"""
        return self.feedback_collector.collect_implicit_feedback(query, response, user_behavior)
    
    def update_from_document_change(self, doc_id: str, old_content: str, new_content: str):
        """Update knowledge graph when documents change"""
        self.context_engine.update_from_document_change(doc_id, old_content, new_content)
    
    def save_knowledge_graph(self, filename: str = "ace_knowledge_graph.pkl"):
        """Save knowledge graph to disk"""
        try:
            import pickle
            data = {
                'knowledge_graph': self.context_engine.knowledge_graph,
                'relationships': self.context_engine.relationships,
                'learning_metrics': self.context_engine.learning_metrics
            }
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")
            return False
    
    def load_knowledge_graph(self, filename: str = "ace_knowledge_graph.pkl"):
        """Load knowledge graph from disk"""
        try:
            import pickle
            import os
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.context_engine.knowledge_graph = data.get('knowledge_graph', {})
                self.context_engine.relationships = data.get('relationships', {})
                self.context_engine.learning_metrics = data.get('learning_metrics', LearningMetrics())
                return True
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
        return False
    
    def _extract_visual_concepts(self, query: str, image_data: Dict) -> List[Dict]:
        """Extract visual concepts from image data for knowledge graph"""
        concepts = []
        
        # Extract basic visual metadata
        if 'filename' in image_data:
            concepts.append({
                "type": "image_metadata",
                "content": f"Image file: {image_data['filename']}"
            })
        
        if 'dimensions' in image_data:
            concepts.append({
                "type": "image_properties", 
                "content": f"Image dimensions: {image_data['dimensions'][0]}x{image_data['dimensions'][1]}"
            })
        
        # Extract query-based visual concepts
        query_lower = query.lower()
        if any(word in query_lower for word in ['error', 'alarm', 'warning', 'problem']):
            concepts.append({
                "type": "troubleshooting",
                "content": f"Visual troubleshooting context for: {query}"
            })
        
        if any(word in query_lower for word in ['equipment', 'device', 'machine', 'system']):
            concepts.append({
                "type": "equipment_analysis",
                "content": f"Equipment analysis context for: {query}"
            })
        
        # Special handling for 4-cells problem
        if any(phrase in query_lower for phrase in ['4cells', '4-cells', '4 cells', 'noisy cells', 'vertical cells']):
            concepts.append({
                "type": "4cells_problem",
                "content": f"4-cells problem analysis: {query} - Check for 4 consecutive vertical noisy cells pattern"
            })
        
        # ECAL/HCAL specific patterns
        if any(word in query_lower for word in ['ecal', 'hcal', 'adc', 'weighted position']):
            concepts.append({
                "type": "ecal_hcal_analysis",
                "content": f"ECAL/HCAL analysis context: {query} - Monitor for 4-cells problem patterns"
            })
        
        return concepts
    
    def save_ace_state(self, path: str):
        """Save complete ACE state"""
        self.context_engine.save_knowledge_graph(path)
        
        # Save additional ACE data
        ace_data = {
            "adaptation_history": self.adaptive_pipeline.adaptation_history,
            "performance_tracker": self.adaptive_pipeline.performance_tracker
        }
        
        with open(f"{path}_ace_system.pkl", 'wb') as f:
            pickle.dump(ace_data, f)
    
    def load_ace_state(self, path: str):
        """Load complete ACE state"""
        success = self.context_engine.load_knowledge_graph(path)
        
        # Load additional ACE data
        try:
            if Path(f"{path}_ace_system.pkl").exists():
                with open(f"{path}_ace_system.pkl", 'rb') as f:
                    ace_data = pickle.load(f)
                
                self.adaptive_pipeline.adaptation_history = ace_data.get("adaptation_history", [])
                self.adaptive_pipeline.performance_tracker = ace_data.get("performance_tracker", {})
        except Exception as e:
            st.error(f"Error loading ACE system state: {str(e)}")
        
        return success

class FeedbackCollector:
    """Collects and manages user feedback"""
    
    def __init__(self):
        self.feedback_history = []
    
    def create_feedback(self, query: str, response: str, user_rating: int,
                       expert_correction: str = None, improvement_suggestions: List[str] = None) -> FeedbackEntry:
        """Create feedback entry"""
        feedback = FeedbackEntry(
            id=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query=query,
            response=response,
            user_rating=user_rating,
            expert_correction=expert_correction,
            timestamp=datetime.now(),
            improvement_suggestions=improvement_suggestions or []
        )
        
        self.feedback_history.append(feedback)
        return feedback

class ACEEvaluationSystem:
    """Evaluates ACE performance and impact"""
    
    def __init__(self):
        self.evaluation_metrics = {}
    
    def evaluate_ace_performance(self, ace_system: ACESystem) -> Dict[str, Any]:
        """Evaluate ACE system performance"""
        metrics = ace_system.get_ace_metrics()
        
        # Calculate performance scores
        accuracy_score = metrics["learning_metrics"]["accuracy_improvement"]
        knowledge_growth = metrics["learning_metrics"]["knowledge_growth_rate"]
        adaptation_frequency = metrics["adaptation_count"] / max(1, metrics["learning_metrics"]["total_interactions"])
        
        return {
            "accuracy_score": accuracy_score,
            "knowledge_growth_rate": knowledge_growth,
            "adaptation_frequency": adaptation_frequency,
            "overall_performance": (accuracy_score + knowledge_growth + adaptation_frequency) / 3,
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics["learning_metrics"]["accuracy_improvement"] < 0.7:
            recommendations.append("Consider collecting more expert feedback to improve accuracy")
        
        if metrics["learning_metrics"]["knowledge_growth_rate"] < 0.1:
            recommendations.append("Increase context diversity to improve knowledge growth")
        
        if metrics["adaptation_count"] < 5:
            recommendations.append("System may need more interactions to trigger meaningful adaptations")
        
        return recommendations


# ============================================================================
# ENHANCED SELF-LEARNING COMPONENTS
# ============================================================================

class DocumentVersionControl:
    """
    PROBLEM: Current system doesn't track document versions or handle updates
    SOLUTION: Implement versioning system with automatic conflict resolution
    """
    
    def __init__(self):
        self.document_versions = {}  # doc_id -> [version1, version2, ...]
        self.version_metadata = {}   # version_id -> metadata
        
    def add_document_version(self, doc_id: str, content: str, metadata: dict):
        """
        Track document versions to enable:
        1. Rollback to previous versions
        2. Compare changes between versions
        3. Identify knowledge drift
        4. Maintain audit trail
        """
        version_id = f"{doc_id}_v{len(self.document_versions.get(doc_id, []))}"
        
        version_data = {
            'version_id': version_id,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata,
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'change_summary': self._generate_change_summary(doc_id, content)
        }
        
        if doc_id not in self.document_versions:
            self.document_versions[doc_id] = []
        
        self.document_versions[doc_id].append(version_data)
        self.version_metadata[version_id] = version_data
        
    def _generate_change_summary(self, doc_id: str, new_content: str) -> dict:
        """
        IMPROVEMENT: Automatically detect and summarize changes
        - What sections were added/removed/modified
        - Severity of changes (minor typo vs major procedure change)
        - Impact on existing knowledge graph
        """
        if doc_id not in self.document_versions or not self.document_versions[doc_id]:
            return {"type": "new_document", "severity": "high"}
        
        old_content = self.document_versions[doc_id][-1]['content']
        
        # Simple change detection (enhance with NLP for production)
        old_words = set(old_content.split())
        new_words = set(new_content.split())
        
        added_words = new_words - old_words
        removed_words = old_words - new_words
        
        change_ratio = len(added_words | removed_words) / max(len(old_words), 1)
        
        return {
            "type": "update",
            "severity": "high" if change_ratio > 0.3 else "medium" if change_ratio > 0.1 else "low",
            "added_words": len(added_words),
            "removed_words": len(removed_words),
            "change_ratio": change_ratio
        }
    
    def should_trigger_relearning(self, doc_id: str) -> bool:
        """
        CRITICAL: Determine if document update requires knowledge graph refresh
        """
        if doc_id not in self.document_versions:
            return False
        
        latest_version = self.document_versions[doc_id][-1]
        change_summary = latest_version.get('change_summary', {})
        
        # Trigger relearning for high-severity changes
        return change_summary.get('severity') in ['high', 'medium']


class DocumentRefreshManager:
    """
    IMPROVEMENT: Automatically determine when documents need updating
    
    Features:
    1. Track query patterns to identify documentation gaps
    2. Monitor low-confidence responses
    3. Suggest document updates based on user feedback
    4. Priority-based refresh scheduling
    """
    
    def __init__(self):
        self.query_patterns = defaultdict(int)
        self.low_confidence_queries = []
        self.refresh_priority = {}
        
    def analyze_query_patterns(self, query: str, response_quality: float):
        """Track which topics are frequently asked about"""
        # Extract key topics from query
        topics = self._extract_topics(query)
        
        for topic in topics:
            self.query_patterns[topic] += 1
            
            # Flag if response quality was low
            if response_quality < 0.6:
                self.low_confidence_queries.append({
                    'query': query,
                    'topic': topic,
                    'timestamp': datetime.now(),
                    'quality': response_quality
                })
    
    def get_refresh_recommendations(self) -> List[dict]:
        """
        CRITICAL: Generate prioritized list of documents to update
        
        Based on:
        1. Frequency of queries about topic
        2. Low-quality responses
        3. Time since last update
        4. User feedback
        """
        recommendations = []
        
        # Analyze patterns
        for topic, count in self.query_patterns.items():
            low_quality_count = sum(
                1 for q in self.low_confidence_queries 
                if q['topic'] == topic
            )
            
            if count > 10 and low_quality_count > 3:
                recommendations.append({
                    'topic': topic,
                    'priority': 'high',
                    'reason': f'{count} queries with {low_quality_count} low-quality responses',
                    'suggested_action': 'Update or create documentation for this topic'
                })
        
        return sorted(recommendations, key=lambda x: self.query_patterns[x['topic']], reverse=True)
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract main topics from query"""
        # Simple keyword extraction (enhance with NLP)
        keywords = ['alarm', 'error', 'restart', 'shutdown', 'cooling', 
                   'pressure', 'temperature', 'calibration', '4-cells', 'noisy', 'vertical']
        
        query_lower = query.lower()
        return [kw for kw in keywords if kw in query_lower]


class EnhancedFeedbackCollector(FeedbackCollector):
    """
    IMPROVEMENTS:
    1. Automatic feedback extraction from usage patterns
    2. Implicit feedback (time spent, follow-up questions)
    3. Confidence calibration
    """
    
    def __init__(self):
        super().__init__()
        self.implicit_feedback = []
        
    def collect_implicit_feedback(self, query: str, response: str, 
                                  user_behavior: dict):
        """
        CRITICAL: Learn from user behavior, not just explicit feedback
        
        Implicit signals:
        1. Did user ask follow-up questions? (response wasn't clear)
        2. Did user copy the response? (response was useful)
        3. How long did user read? (engagement level)
        4. Did user provide alternative phrasing? (understanding issues)
        """
        implicit_rating = 3  # Neutral default
        
        # Adjust based on behavior
        if user_behavior.get('follow_up_questions', 0) > 2:
            implicit_rating -= 1  # Response wasn't clear
        
        if user_behavior.get('response_copied', False):
            implicit_rating += 1  # Response was useful
        
        if user_behavior.get('read_time_seconds', 0) > 30:
            implicit_rating += 0.5  # High engagement
        
        self.implicit_feedback.append({
            'query': query,
            'response': response,
            'implicit_rating': implicit_rating,
            'behavior': user_behavior,
            'timestamp': datetime.now()
        })
        
        return implicit_rating


class MultiModalLearningEngine:
    """
    IMPROVEMENT: Learn from both text AND vision feedback simultaneously
    
    Key features:
    1. Cross-modal knowledge transfer
    2. Visual pattern recognition learning
    3. Image-text alignment improvements
    """
    
    def __init__(self):
        self.visual_patterns = {}
        self.text_visual_mappings = defaultdict(list)
        
    def learn_from_vision_feedback(self, image_data: dict, analysis: str, 
                                   expert_correction: str):
        """
        CRITICAL: Extract visual patterns from expert corrections
        
        Example: If expert says "These are 4 noisy cells in vertical pattern",
        learn to recognize this pattern in future images
        """
        # Extract visual features (simplified - use CV in production)
        visual_signature = self._create_visual_signature(image_data)
        
        # Store pattern with expert label
        pattern_id = hashlib.md5(visual_signature.encode()).hexdigest()[:8]
        self.visual_patterns[pattern_id] = {
            'signature': visual_signature,
            'expert_label': expert_correction,
            'confidence': 0.9,
            'examples': [image_data['filename']],
            'learned_at': datetime.now()
        }
        
        # Map text descriptions to visual patterns
        text_keywords = self._extract_visual_keywords(expert_correction)
        for keyword in text_keywords:
            self.text_visual_mappings[keyword].append(pattern_id)
    
    def _create_visual_signature(self, image_data: dict) -> str:
        """
        Create a signature for visual pattern matching
        In production, use:
        1. Image embeddings (CLIP, ViT)
        2. Perceptual hashing
        3. Feature extraction (SIFT, SURF)
        """
        # Simplified signature based on metadata
        return f"{image_data['dimensions']}_{image_data.get('format', 'unknown')}"
    
    def _extract_visual_keywords(self, text: str) -> List[str]:
        """Extract visual descriptors from text"""
        visual_keywords = ['noisy', 'vertical', 'pattern', 'cells', 
                          'consecutive', 'horizontal', 'bright', 'dark']
        
        text_lower = text.lower()
        return [kw for kw in visual_keywords if kw in text_lower]


class EnhancedContextEvolutionEngine(ContextEvolutionEngine):
    """
    IMPROVEMENTS:
    1. Automatic knowledge pruning based on usage
    2. Confidence decay for outdated information
    3. Relationship strength tracking
    4. Contradiction detection and resolution
    """
    
    def __init__(self):
        super().__init__()
        self.relationship_strength = defaultdict(lambda: defaultdict(float))
        self.contradiction_log = []
        self.document_version_control = DocumentVersionControl()
        
    def update_from_document_change(self, doc_id: str, old_content: str, new_content: str):
        """
        CRITICAL IMPROVEMENT: Automatically update knowledge graph when documents change
        
        This is the KEY to true self-learning:
        1. Identify affected knowledge nodes
        2. Update or deprecate outdated information
        3. Create new nodes for new information
        4. Maintain knowledge integrity
        """
        # Find nodes related to this document
        affected_nodes = self._find_nodes_by_source(doc_id)
        
        # Extract new knowledge from updated content
        new_knowledge = self._extract_knowledge(new_content)
        old_knowledge = self._extract_knowledge(old_content)
        
        # Identify what changed
        deprecated_knowledge = set(old_knowledge) - set(new_knowledge)
        new_knowledge_items = set(new_knowledge) - set(old_knowledge)
        
        # Deprecate outdated nodes
        for node_id in affected_nodes:
            node = self.knowledge_graph[node_id]
            if any(dep in node.content for dep in deprecated_knowledge):
                # Reduce confidence instead of deleting (for audit trail)
                node.confidence *= 0.5
                node.category = f"deprecated_{node.category}"
                self.contradiction_log.append({
                    'timestamp': datetime.now(),
                    'node_id': node_id,
                    'reason': 'document_update',
                    'old_content': node.content
                })
        
        # Add new knowledge nodes
        for knowledge_item in new_knowledge_items:
            self.add_context_node(
                content=knowledge_item,
                category="updated_knowledge",
                source=f"document_update_{doc_id}",
                confidence=0.9  # High confidence for recent updates
            )
    
    def _find_nodes_by_source(self, source: str) -> List[str]:
        """Find all nodes created from a specific source"""
        return [
            node_id for node_id, node in self.knowledge_graph.items()
            if source in node.source
        ]
    
    def _extract_knowledge(self, content: str) -> List[str]:
        """
        IMPROVEMENT NEEDED: Use NLP to extract structured knowledge
        
        Current implementation is basic. For production, consider:
        1. Named Entity Recognition (NER)
        2. Relationship extraction
        3. Fact verification
        4. Semantic understanding
        """
        # Simple sentence-based extraction
        sentences = content.split('. ')
        return [s.strip() for s in sentences if len(s.split()) > 5]
    
    def detect_contradictions(self, new_node: ContextNode) -> List[str]:
        """
        CRITICAL: Detect when new information contradicts existing knowledge
        
        This enables:
        1. Alert operators to conflicting information
        2. Trigger expert review
        3. Maintain knowledge integrity
        """
        contradictions = []
        new_words = set(new_node.content.lower().split())
        
        for node_id, node in self.knowledge_graph.items():
            if node.category == new_node.category:
                # Check for opposite meanings (simplified)
                node_words = set(node.content.lower().split())
                
                # Look for negation patterns
                negation_indicators = ['not', 'never', 'dont', "don't", 'avoid', 'instead']
                
                if (len(new_words.intersection(node_words)) > 5 and
                    any(neg in new_words for neg in negation_indicators) !=
                    any(neg in node_words for neg in negation_indicators)):
                    
                    contradictions.append(node_id)
                    self.contradiction_log.append({
                        'timestamp': datetime.now(),
                        'existing_node': node_id,
                        'new_node': new_node.id,
                        'confidence_diff': abs(node.confidence - new_node.confidence)
                    })
        
        return contradictions
    
    def apply_confidence_decay(self, days_threshold: int = 30):
        """
        IMPROVEMENT: Automatically reduce confidence in old, unused knowledge
        
        This prevents stale information from affecting decisions
        """
        current_time = datetime.now()
        
        for node_id, node in self.knowledge_graph.items():
            if node.last_accessed:
                days_since_access = (current_time - node.last_accessed).days
                
                if days_since_access > days_threshold:
                    # Apply exponential decay
                    decay_factor = np.exp(-days_since_access / days_threshold)
                    node.confidence *= decay_factor
                    
                    # Mark for review if confidence drops too low
                    if node.confidence < 0.3:
                        node.category = f"review_needed_{node.category}"
