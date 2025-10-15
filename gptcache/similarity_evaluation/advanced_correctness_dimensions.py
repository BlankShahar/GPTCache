"""
Advanced Correctness Dimension Evaluators
=========================================

Individual evaluators for each dimension of the Advanced Correctness framework.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class SemanticEquivalenceEvaluator:
    """
    Evaluates semantic equivalence between queries.
    
    Uses embeddings and/or text similarity to determine if two queries
    have the same underlying meaning and intent.
    """
    
    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
    
    def evaluate(
        self,
        src_question: str,
        cache_question: str,
        src_embedding: Optional[np.ndarray] = None,
        cache_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate semantic equivalence between two questions.
        
        Returns:
            float: Score from 0.0 (completely different) to 1.0 (semantically identical)
        """
        
        # Exact match gets perfect score
        if src_question.lower().strip() == cache_question.lower().strip():
            return 1.0
        
        # Use embeddings if available
        if src_embedding is not None and cache_embedding is not None:
            return self._embedding_similarity(src_embedding, cache_embedding)
        
        # Fallback to text-based similarity
        return self._text_similarity(src_question, cache_question)
    
    def _embedding_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        # Ensure embeddings are numpy arrays
        emb1 = np.array(embedding1).flatten()
        emb2 = np.array(embedding2).flatten()
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return (similarity + 1.0) / 2.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text-based similarity using token overlap."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0


class FactualConsistencyEvaluator:
    """
    Evaluates factual consistency between queries and responses.
    
    Checks if key entities, numbers, dates, and relationships are consistent.
    """
    
    def __init__(self):
        # Patterns for entity extraction
        self.number_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')
        self.date_pattern = re.compile(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', re.IGNORECASE)
        self.capitalized_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
    
    def evaluate(
        self,
        src_question: str,
        cache_question: str,
        cache_answer: str
    ) -> float:
        """
        Evaluate factual consistency.
        
        Returns:
            float: Score from 0.0 (factually inconsistent) to 1.0 (fully consistent)
        """
        
        # Extract entities from questions
        src_entities = self._extract_entities(src_question)
        cache_entities = self._extract_entities(cache_question)
        
        # Check if critical entities match
        entity_score = self._compare_entities(src_entities, cache_entities)
        
        # Check for contradictory facts
        contradiction_penalty = self._check_contradictions(
            src_question, cache_question, cache_answer
        )
        
        final_score = entity_score * (1.0 - contradiction_penalty)
        
        return max(0.0, min(1.0, final_score))
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities from text."""
        return {
            'numbers': self.number_pattern.findall(text),
            'dates': self.date_pattern.findall(text),
            'proper_nouns': self.capitalized_pattern.findall(text)
        }
    
    def _compare_entities(
        self,
        entities1: Dict[str, List[str]],
        entities2: Dict[str, List[str]]
    ) -> float:
        """Compare entity sets for consistency."""
        
        scores = []
        
        for entity_type in ['numbers', 'dates', 'proper_nouns']:
            set1 = set(entities1.get(entity_type, []))
            set2 = set(entities2.get(entity_type, []))
            
            # If both empty, perfect match
            if not set1 and not set2:
                continue
            
            # If one has entities and other doesn't, potential issue
            if (set1 and not set2) or (set2 and not set1):
                # But might be acceptable (paraphrasing)
                scores.append(0.7)
                continue
            
            # Calculate overlap
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            
            if union:
                scores.append(len(intersection) / len(union))
            else:
                scores.append(1.0)
        
        return sum(scores) / len(scores) if scores else 1.0
    
    def _check_contradictions(
        self,
        src_question: str,
        cache_question: str,
        cache_answer: str
    ) -> float:
        """Check for obvious contradictions."""
        
        # Simple heuristics for contradiction detection
        contradiction_keywords = [
            ('city', 'state'), ('state', 'country'),
            ('iPhone', 'Android'), ('first', 'last'),
            ('largest', 'smallest'), ('minimum', 'maximum')
        ]
        
        src_lower = src_question.lower()
        cache_lower = cache_question.lower()
        
        penalty = 0.0
        
        for word1, word2 in contradiction_keywords:
            if word1 in src_lower and word2 in cache_lower:
                penalty += 0.3
            elif word2 in src_lower and word1 in cache_lower:
                penalty += 0.3
        
        return min(1.0, penalty)


class ContextualRelevanceEvaluator:
    """
    Evaluates contextual relevance in multi-turn conversations.
    
    Assesses whether cached responses appropriately use conversation history.
    """
    
    def __init__(self):
        self.context_indicators = [
            'it', 'its', 'they', 'them', 'their', 'this', 'that', 
            'these', 'those', 'there', 'here'
        ]
    
    def evaluate(
        self,
        src_question: str,
        cache_question: str,
        src_context: Dict[str, Any],
        cache_context: Dict[str, Any]
    ) -> float:
        """
        Evaluate contextual relevance.
        
        Returns:
            float: Score from 0.0 (wrong context) to 1.0 (perfect context match)
        """
        
        # Check if question has context indicators
        has_context_dependency = self._has_context_dependency(src_question)
        
        if not has_context_dependency:
            # No context dependency, so context matching is less critical
            return 1.0
        
        # If query needs context, check if contexts align
        if not src_context and not cache_context:
            # Both lack context when needed - problematic
            return 0.5
        
        # Compare context keys
        context_score = self._compare_contexts(src_context, cache_context)
        
        return context_score
    
    def _has_context_dependency(self, question: str) -> bool:
        """Check if question depends on previous context."""
        question_lower = question.lower()
        
        # Check for pronouns and demonstratives
        for indicator in self.context_indicators:
            if f' {indicator} ' in f' {question_lower} ':
                return True
        
        # Check if question is short (likely a follow-up)
        if len(question.split()) < 5:
            return True
        
        return False
    
    def _compare_contexts(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Compare two context dictionaries."""
        
        if not context1 or not context2:
            return 0.5
        
        # Check for matching previous query/topic
        prev_query1 = context1.get('previous_query', '').lower()
        prev_query2 = context2.get('previous_query', '').lower()
        
        if prev_query1 and prev_query2:
            # Simple token overlap
            tokens1 = set(prev_query1.split())
            tokens2 = set(prev_query2.split())
            
            if tokens1 and tokens2:
                overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
                return overlap
        
        return 0.5


class ToneStyleEvaluator:
    """
    Evaluates tone and style consistency.
    
    Assesses whether the cached response maintains appropriate formality and voice.
    """
    
    def __init__(self):
        self.formal_indicators = [
            'please', 'kindly', 'would', 'could', 'may', 'shall',
            'furthermore', 'therefore', 'consequently', 'thus'
        ]
        self.informal_indicators = [
            "what's", "how's", "can't", "won't", "don't",
            'yeah', 'gonna', 'wanna', 'gotta', 'kinda'
        ]
        self.technical_indicators = [
            'algorithm', 'function', 'parameter', 'implementation',
            'architecture', 'framework', 'protocol', 'syntax'
        ]
    
    def evaluate(
        self,
        src_question: str,
        cache_question: str,
        cache_answer: str
    ) -> float:
        """
        Evaluate tone and style preservation.
        
        Returns:
            float: Score from 0.0 (style mismatch) to 1.0 (perfect style match)
        """
        
        src_style = self._analyze_style(src_question)
        cache_style = self._analyze_style(cache_question)
        
        # Compare formality levels
        formality_diff = abs(src_style['formality'] - cache_style['formality'])
        formality_score = 1.0 - (formality_diff / 2.0)  # Normalize
        
        # Compare technical level
        technical_diff = abs(src_style['technicality'] - cache_style['technicality'])
        technical_score = 1.0 - (technical_diff / 2.0)
        
        # Average the scores
        return (formality_score + technical_score) / 2.0
    
    def _analyze_style(self, text: str) -> Dict[str, float]:
        """Analyze text style characteristics."""
        text_lower = text.lower()
        tokens = text_lower.split()
        
        if not tokens:
            return {'formality': 0.5, 'technicality': 0.5}
        
        # Calculate formality (0=informal, 1=formal)
        formal_count = sum(1 for word in self.formal_indicators if word in text_lower)
        informal_count = sum(1 for word in self.informal_indicators if word in text_lower)
        
        if formal_count + informal_count > 0:
            formality = formal_count / (formal_count + informal_count)
        else:
            # Default to neutral
            formality = 0.5
        
        # Calculate technicality
        technical_count = sum(1 for word in self.technical_indicators if word in text_lower)
        technicality = min(1.0, technical_count / max(1, len(tokens) * 0.1))
        
        return {
            'formality': formality,
            'technicality': technicality
        }


class InstructionAdherenceEvaluator:
    """
    Evaluates instruction adherence.
    
    Checks if cached responses follow format requirements and constraints.
    """
    
    def __init__(self):
        self.format_keywords = {
            'list': ['list', 'enumerate', 'bullet', 'points'],
            'summary': ['summarize', 'summary', 'brief', 'overview'],
            'explain': ['explain', 'describe', 'tell', 'what is'],
            'steps': ['steps', 'how to', 'instructions', 'guide'],
            'compare': ['compare', 'difference', 'versus', 'vs'],
            'example': ['example', 'instance', 'sample', 'demonstration']
        }
        
        self.number_pattern = re.compile(r'\b(\d+)\s+(?:items?|points?|steps?|examples?)\b', re.IGNORECASE)
    
    def evaluate(
        self,
        src_question: str,
        cache_question: str,
        cache_answer: str
    ) -> float:
        """
        Evaluate instruction adherence.
        
        Returns:
            float: Score from 0.0 (doesn't follow instructions) to 1.0 (perfect adherence)
        """
        
        src_instructions = self._extract_instructions(src_question)
        cache_instructions = self._extract_instructions(cache_question)
        
        # If no specific instructions, format is flexible
        if not src_instructions['format'] and not src_instructions['count']:
            return 1.0
        
        # Compare format requirements
        format_score = self._compare_formats(
            src_instructions['format'],
            cache_instructions['format']
        )
        
        # Compare count requirements
        count_score = self._compare_counts(
            src_instructions['count'],
            cache_instructions['count']
        )
        
        # Both must match for high score
        return min(format_score, count_score)
    
    def _extract_instructions(self, text: str) -> Dict[str, Any]:
        """Extract format and count instructions from text."""
        text_lower = text.lower()
        
        # Detect format type
        detected_format = None
        for format_type, keywords in self.format_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_format = format_type
                break
        
        # Detect count requirements
        count_match = self.number_pattern.search(text_lower)
        detected_count = int(count_match.group(1)) if count_match else None
        
        return {
            'format': detected_format,
            'count': detected_count
        }
    
    def _compare_formats(
        self,
        format1: Optional[str],
        format2: Optional[str]
    ) -> float:
        """Compare format requirements."""
        if format1 is None and format2 is None:
            return 1.0
        
        if format1 == format2:
            return 1.0
        
        # Different format requirements - problematic
        if format1 and format2 and format1 != format2:
            return 0.3
        
        # One has format requirement, other doesn't
        return 0.6
    
    def _compare_counts(
        self,
        count1: Optional[int],
        count2: Optional[int]
    ) -> float:
        """Compare count requirements."""
        if count1 is None and count2 is None:
            return 1.0
        
        if count1 == count2:
            return 1.0
        
        # Different counts - problematic
        if count1 and count2:
            # Allow small differences
            diff = abs(count1 - count2)
            if diff <= 1:
                return 0.8
            elif diff <= 2:
                return 0.5
            else:
                return 0.2
        
        # One has count, other doesn't
        return 0.6


