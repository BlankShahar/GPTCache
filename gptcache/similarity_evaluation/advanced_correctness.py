"""
Advanced Correctness Evaluation Module
=======================================

Implements multi-dimensional correctness evaluation for semantic caching
that goes beyond simple similarity matching.

The AdvancedCorrectnessEvaluation evaluates cache hits across 5 dimensions:
1. Semantic Equivalence - Meaning preservation
2. Factual Consistency - Information accuracy
3. Contextual Relevance - Conversation context integration
4. Tone & Style Preservation - Voice and formality consistency
5. Instruction Adherence - Format and constraint compliance
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from gptcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation


class AdvancedCorrectnessEvaluation(SimilarityEvaluation):
    """
    Multi-dimensional correctness evaluation for advanced semantic caching.
    
    This evaluator assesses cache hits across five quality dimensions,
    returning the minimum score to ensure no critical aspect is compromised.
    
    Args:
        semantic_weight (float): Weight for semantic equivalence (0.0-1.0)
        factual_weight (float): Weight for factual consistency (0.0-1.0)
        contextual_weight (float): Weight for contextual relevance (0.0-1.0)
        tone_weight (float): Weight for tone/style preservation (0.0-1.0)
        instruction_weight (float): Weight for instruction adherence (0.0-1.0)
        use_minimum (bool): If True, use MIN of scores; if False, use weighted average
        threshold (float): Minimum acceptable score (default 0.7)
    
    Example:
        .. code-block:: python
        
            from gptcache import cache
            from gptcache.similarity_evaluation.advanced_correctness import (
                AdvancedCorrectnessEvaluation
            )
            
            evaluator = AdvancedCorrectnessEvaluation(
                semantic_weight=1.0,
                factual_weight=1.0,
                contextual_weight=0.8,
                use_minimum=True
            )
            
            cache.init(similarity_evaluation=evaluator)
    """
    
    def __init__(
        self,
        semantic_weight: float = 1.0,
        factual_weight: float = 1.0,
        contextual_weight: float = 1.0,
        tone_weight: float = 1.0,
        instruction_weight: float = 1.0,
        use_minimum: bool = True,
        threshold: float = 0.7,
        embedding_model: Optional[Any] = None,
        enable_factual_check: bool = True,
        enable_context_tracking: bool = True,
        enable_instruction_parsing: bool = True
    ):
        self.semantic_weight = semantic_weight
        self.factual_weight = factual_weight
        self.contextual_weight = contextual_weight
        self.tone_weight = tone_weight
        self.instruction_weight = instruction_weight
        self.use_minimum = use_minimum
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.enable_factual_check = enable_factual_check
        self.enable_context_tracking = enable_context_tracking
        self.enable_instruction_parsing = enable_instruction_parsing
        
        # Initialize dimension evaluators
        from gptcache.similarity_evaluation.advanced_correctness_dimensions import (
            SemanticEquivalenceEvaluator,
            FactualConsistencyEvaluator,
            ContextualRelevanceEvaluator,
            ToneStyleEvaluator,
            InstructionAdherenceEvaluator
        )
        
        self.semantic_evaluator = SemanticEquivalenceEvaluator(embedding_model)
        self.factual_evaluator = FactualConsistencyEvaluator() if enable_factual_check else None
        self.context_evaluator = ContextualRelevanceEvaluator() if enable_context_tracking else None
        self.tone_evaluator = ToneStyleEvaluator()
        self.instruction_evaluator = InstructionAdherenceEvaluator() if enable_instruction_parsing else None
    
    def evaluation(
        self, 
        src_dict: Dict[str, Any], 
        cache_dict: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Evaluate cache correctness across all applicable dimensions.
        
        Args:
            src_dict: Source query dictionary containing:
                - 'question': The current query text
                - 'embedding': Optional pre-computed embedding
                - 'context': Optional conversation context
            cache_dict: Cached data dictionary containing:
                - 'question': The cached query text
                - 'answer': The cached response
                - 'embedding': Optional cached embedding
                - 'metadata': Optional cached metadata
            **kwargs: Additional parameters
        
        Returns:
            float: Overall correctness score (0.0-1.0)
        """
        
        scores = []
        weights = []
        
        # 1. Semantic Equivalence
        if self.semantic_weight > 0:
            semantic_score = self._evaluate_semantic(src_dict, cache_dict)
            scores.append(semantic_score)
            weights.append(self.semantic_weight)
        
        # 2. Factual Consistency
        if self.factual_weight > 0 and self.factual_evaluator:
            factual_score = self._evaluate_factual(src_dict, cache_dict)
            scores.append(factual_score)
            weights.append(self.factual_weight)
        
        # 3. Contextual Relevance
        if self.contextual_weight > 0 and self.context_evaluator:
            context_score = self._evaluate_context(src_dict, cache_dict)
            scores.append(context_score)
            weights.append(self.contextual_weight)
        
        # 4. Tone & Style Preservation
        if self.tone_weight > 0:
            tone_score = self._evaluate_tone(src_dict, cache_dict)
            scores.append(tone_score)
            weights.append(self.tone_weight)
        
        # 5. Instruction Adherence
        if self.instruction_weight > 0 and self.instruction_evaluator:
            instruction_score = self._evaluate_instruction(src_dict, cache_dict)
            scores.append(instruction_score)
            weights.append(self.instruction_weight)
        
        # Calculate overall score
        if self.use_minimum:
            # Use minimum score to ensure no dimension falls below threshold
            overall_score = min(scores) if scores else 0.0
        else:
            # Use weighted average
            if scores and weights:
                overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                overall_score = 0.0
        
        return overall_score
    
    def _evaluate_semantic(self, src_dict: Dict, cache_dict: Dict) -> float:
        """Evaluate semantic equivalence dimension."""
        return self.semantic_evaluator.evaluate(
            src_dict.get('question', ''),
            cache_dict.get('question', ''),
            src_dict.get('embedding'),
            cache_dict.get('embedding')
        )
    
    def _evaluate_factual(self, src_dict: Dict, cache_dict: Dict) -> float:
        """Evaluate factual consistency dimension."""
        if not self.factual_evaluator:
            return 1.0
        
        return self.factual_evaluator.evaluate(
            src_dict.get('question', ''),
            cache_dict.get('question', ''),
            cache_dict.get('answer', '')
        )
    
    def _evaluate_context(self, src_dict: Dict, cache_dict: Dict) -> float:
        """Evaluate contextual relevance dimension."""
        if not self.context_evaluator:
            return 1.0
        
        return self.context_evaluator.evaluate(
            src_dict.get('question', ''),
            cache_dict.get('question', ''),
            src_dict.get('context', {}),
            cache_dict.get('context', {})
        )
    
    def _evaluate_tone(self, src_dict: Dict, cache_dict: Dict) -> float:
        """Evaluate tone and style preservation dimension."""
        return self.tone_evaluator.evaluate(
            src_dict.get('question', ''),
            cache_dict.get('question', ''),
            cache_dict.get('answer', '')
        )
    
    def _evaluate_instruction(self, src_dict: Dict, cache_dict: Dict) -> float:
        """Evaluate instruction adherence dimension."""
        if not self.instruction_evaluator:
            return 1.0
        
        return self.instruction_evaluator.evaluate(
            src_dict.get('question', ''),
            cache_dict.get('question', ''),
            cache_dict.get('answer', '')
        )
    
    def range(self) -> Tuple[float, float]:
        """
        Return the range of similarity scores.
        
        Returns:
            Tuple[float, float]: (minimum=0.0, maximum=1.0)
        """
        return 0.0, 1.0
    
    def get_dimension_scores(
        self,
        src_dict: Dict[str, Any],
        cache_dict: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed scores for each dimension (for debugging/analysis).
        
        Returns:
            Dict mapping dimension names to scores
        """
        return {
            'semantic_equivalence': self._evaluate_semantic(src_dict, cache_dict),
            'factual_consistency': self._evaluate_factual(src_dict, cache_dict),
            'contextual_relevance': self._evaluate_context(src_dict, cache_dict),
            'tone_style_preservation': self._evaluate_tone(src_dict, cache_dict),
            'instruction_adherence': self._evaluate_instruction(src_dict, cache_dict),
        }


