"""
Advanced Correctness Evaluation Example
========================================

This example demonstrates how to use the Advanced Correctness evaluation
with GPTCache to implement multi-dimensional cache quality assessment.
"""

import os
from gptcache import cache
from gptcache.adapter import openai
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation import AdvancedCorrectnessEvaluation


def run_basic_example():
    """
    Basic example using Advanced Correctness Evaluation.
    """
    print("=" * 60)
    print("BASIC ADVANCED CORRECTNESS EXAMPLE")
    print("=" * 60)
    
    # Initialize embedding
    embedding_onnx = EmbeddingOnnx()
    
    # Create Advanced Correctness Evaluator
    # Using all 5 dimensions with equal weights
    evaluator = AdvancedCorrectnessEvaluation(
        semantic_weight=1.0,
        factual_weight=1.0,
        contextual_weight=1.0,
        tone_weight=1.0,
        instruction_weight=1.0,
        use_minimum=True,  # Use MIN to ensure all dimensions pass
        threshold=0.7
    )
    
    # Initialize cache
    data_manager = get_data_manager(
        CacheBase("sqlite"),
        VectorBase("faiss", dimension=embedding_onnx.dimension)
    )
    
    cache.init(
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluator
    )
    
    # Set OpenAI key (make sure it's set in environment)
    cache.set_openai_key()
    
    # Test paraphrased queries (should hit cache)
    print("\n1. Testing Paraphrased Queries (should cache hit)")
    print("-" * 60)
    
    question1 = "What's the capital of France?"
    print(f"Query 1: {question1}")
    response1 = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': question1}]
    )
    print(f"Response 1: {openai.get_message_from_openai_answer(response1)}\n")
    
    question2 = "Which city serves as France's capital?"
    print(f"Query 2 (paraphrase): {question2}")
    response2 = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': question2}]
    )
    print(f"Response 2: {openai.get_message_from_openai_answer(response2)}")
    print(f"Cache hit expected: YES (semantic equivalence preserved)")
    
    # Test factually different queries (should miss cache)
    print("\n2. Testing Factually Different Queries (should cache miss)")
    print("-" * 60)
    
    question3 = "What's the population of New York City?"
    print(f"Query 3: {question3}")
    response3 = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': question3}]
    )
    print(f"Response 3: {openai.get_message_from_openai_answer(response3)}\n")
    
    question4 = "What's the population of New York State?"
    print(f"Query 4 (different fact): {question4}")
    response4 = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': question4}]
    )
    print(f"Response 4: {openai.get_message_from_openai_answer(response4)}")
    print(f"Cache hit expected: NO (factual inconsistency)")
    
    print("\n" + "=" * 60)
    print(f"Cache report:")
    print(f"  Hit count: {cache.data_manager.hit_cache_callback.hit_count if hasattr(cache.data_manager.hit_cache_callback, 'hit_count') else 'N/A'}")
    print("=" * 60)


def run_weighted_example():
    """
    Example using weighted dimension importance.
    """
    print("\n\n" + "=" * 60)
    print("WEIGHTED DIMENSIONS EXAMPLE")
    print("=" * 60)
    
    embedding_onnx = EmbeddingOnnx()
    
    # Create evaluator with custom weights
    # Prioritize semantic and factual accuracy over other dimensions
    evaluator = AdvancedCorrectnessEvaluation(
        semantic_weight=2.0,      # Double weight on semantic
        factual_weight=2.0,       # Double weight on factual
        contextual_weight=1.0,
        tone_weight=0.5,          # Less weight on tone
        instruction_weight=1.0,
        use_minimum=False,  # Use weighted average instead of MIN
        threshold=0.75
    )
    
    data_manager = get_data_manager(
        CacheBase("sqlite", sql_url="sqlite:///./weighted_cache.db"),
        VectorBase("faiss", dimension=embedding_onnx.dimension)
    )
    
    cache.init(
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluator
    )
    
    print("\nUsing weighted evaluation:")
    print("  Semantic: 2.0x")
    print("  Factual: 2.0x")
    print("  Contextual: 1.0x")
    print("  Tone: 0.5x")
    print("  Instruction: 1.0x")
    print("\nThis configuration is ideal for fact-heavy applications")
    print("where semantic accuracy matters more than tone.")


def run_custom_configuration():
    """
    Example with selective dimension evaluation.
    """
    print("\n\n" + "=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    embedding_onnx = EmbeddingOnnx()
    
    # Create evaluator with some dimensions disabled
    evaluator = AdvancedCorrectnessEvaluation(
        semantic_weight=1.0,
        factual_weight=1.0,
        contextual_weight=0.0,    # Disable context checking
        tone_weight=0.0,          # Disable tone checking
        instruction_weight=1.0,
        use_minimum=True,
        enable_factual_check=True,
        enable_context_tracking=False,  # Disable context tracking
        enable_instruction_parsing=True
    )
    
    data_manager = get_data_manager(
        CacheBase("sqlite", sql_url="sqlite:///./custom_cache.db"),
        VectorBase("faiss", dimension=embedding_onnx.dimension)
    )
    
    cache.init(
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluator
    )
    
    print("\nUsing custom configuration:")
    print("  ✓ Semantic Equivalence: Enabled")
    print("  ✓ Factual Consistency: Enabled")
    print("  ✗ Contextual Relevance: Disabled")
    print("  ✗ Tone & Style: Disabled")
    print("  ✓ Instruction Adherence: Enabled")
    print("\nThis configuration is ideal for stateless Q&A applications")
    print("where context is not important.")


def demonstrate_dimension_scores():
    """
    Example showing how to get detailed dimension scores.
    """
    print("\n\n" + "=" * 60)
    print("DIMENSION SCORE DEBUGGING EXAMPLE")
    print("=" * 60)
    
    from gptcache.similarity_evaluation.advanced_correctness import AdvancedCorrectnessEvaluation
    
    evaluator = AdvancedCorrectnessEvaluation()
    
    # Test queries
    src_dict = {
        'question': 'What are the benefits of exercise?',
        'embedding': None
    }
    
    cache_dict = {
        'question': 'What are the advantages of working out?',
        'answer': 'Exercise improves health, boosts mood, and increases energy.',
        'embedding': None
    }
    
    # Get overall score
    overall_score = evaluator.evaluation(src_dict, cache_dict)
    
    # Get detailed dimension scores
    dimension_scores = evaluator.get_dimension_scores(src_dict, cache_dict)
    
    print("\nQuery Comparison:")
    print(f"  Source: {src_dict['question']}")
    print(f"  Cache:  {cache_dict['question']}")
    print(f"\nOverall Score: {overall_score:.3f}")
    print(f"\nDimension Breakdown:")
    for dimension, score in dimension_scores.items():
        status = "✓" if score >= 0.7 else "✗"
        print(f"  {status} {dimension:25s}: {score:.3f}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Some examples will not work.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning dimension score debugging example only...\n")
        demonstrate_dimension_scores()
    else:
        # Run all examples
        run_basic_example()
        run_weighted_example()
        run_custom_configuration()
        demonstrate_dimension_scores()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


