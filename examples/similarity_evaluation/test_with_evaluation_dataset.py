"""
Test Advanced Correctness with Evaluation Dataset
=================================================

This script tests the Advanced Correctness implementation against
the comprehensive evaluation dataset we created.
"""

import json
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gptcache.similarity_evaluation.advanced_correctness import AdvancedCorrectnessEvaluation
from gptcache.embedding import Onnx as EmbeddingOnnx


class AdvancedCorrectnessValidator:
    """Validates Advanced Correctness implementation against test dataset."""
    
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.embedding_model = EmbeddingOnnx()
        self.evaluator = AdvancedCorrectnessEvaluation(
            embedding_model=self.embedding_model
        )
        
        self.results = {
            'paraphrased_pairs': {'passed': 0, 'failed': 0, 'details': []},
            'context_dependent_sequences': {'passed': 0, 'failed': 0, 'details': []},
            'subtle_factual_variations': {'passed': 0, 'failed': 0, 'details': []},
            'instructional_variations': {'passed': 0, 'failed': 0, 'details': []},
            'true_negatives': {'passed': 0, 'failed': 0, 'details': []}
        }
    
    def test_paraphrased_pairs(self):
        """Test semantic equivalence on paraphrased queries."""
        print("\n" + "=" * 60)
        print("TESTING: Paraphrased Pairs (Semantic Equivalence)")
        print("=" * 60)
        
        category_entries = [e for e in self.dataset['test_entries'] 
                           if e['category'] == 'paraphrased_pairs']
        
        for entry in category_entries:
            test_id = entry['test_id']
            original_query = entry['query']
            paraphrase_query = entry['paraphrase_query']
            
            print(f"\nTest {test_id}:")
            print(f"  Original: {original_query}")
            print(f"  Paraphrase: {paraphrase_query}")
            
            # Get embeddings
            original_emb = self.embedding_model.to_embeddings(original_query)
            paraphrase_emb = self.embedding_model.to_embeddings(paraphrase_query)
            
            # Evaluate
            src_dict = {'question': paraphrase_query, 'embedding': paraphrase_emb}
            cache_dict = {
                'question': original_query,
                'answer': entry['ground_truth_answer'],
                'embedding': original_emb
            }
            
            score = self.evaluator.evaluation(src_dict, cache_dict)
            dimension_scores = self.evaluator.get_dimension_scores(src_dict, cache_dict)
            
            # Check if semantic equivalence is high
            passed = dimension_scores['semantic_equivalence'] >= 0.7
            
            print(f"  Overall Score: {score:.3f}")
            print(f"  Semantic Score: {dimension_scores['semantic_equivalence']:.3f}")
            print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
            
            if passed:
                self.results['paraphrased_pairs']['passed'] += 1
            else:
                self.results['paraphrased_pairs']['failed'] += 1
                self.results['paraphrased_pairs']['details'].append({
                    'test_id': test_id,
                    'score': score,
                    'dimension_scores': dimension_scores
                })
    
    def test_factual_variations(self):
        """Test factual discrimination."""
        print("\n" + "=" * 60)
        print("TESTING: Subtle Factual Variations (Factual Consistency)")
        print("=" * 60)
        
        category_entries = [e for e in self.dataset['test_entries']
                           if e['category'] == 'subtle_factual_variations']
        
        for entry in category_entries:
            test_id = entry['test_id']
            original_query = entry['query']
            similar_query = entry['similar_query']
            
            print(f"\nTest {test_id}:")
            print(f"  Original: {original_query}")
            print(f"  Similar (but different facts): {similar_query}")
            
            # Get embeddings
            original_emb = self.embedding_model.to_embeddings(original_query)
            similar_emb = self.embedding_model.to_embeddings(similar_query)
            
            # Evaluate
            src_dict = {'question': similar_query, 'embedding': similar_emb}
            cache_dict = {
                'question': original_query,
                'answer': entry['ground_truth_answer'],
                'embedding': original_emb
            }
            
            score = self.evaluator.evaluation(src_dict, cache_dict)
            dimension_scores = self.evaluator.get_dimension_scores(src_dict, cache_dict)
            
            # Should have lower factual consistency score
            passed = dimension_scores['factual_consistency'] < 0.7 or score < 0.7
            
            print(f"  Overall Score: {score:.3f}")
            print(f"  Factual Score: {dimension_scores['factual_consistency']:.3f}")
            print(f"  Result: {'✓ PASS (correctly detected difference)' if passed else '✗ FAIL (missed difference)'}")
            
            if passed:
                self.results['subtle_factual_variations']['passed'] += 1
            else:
                self.results['subtle_factual_variations']['failed'] += 1
                self.results['subtle_factual_variations']['details'].append({
                    'test_id': test_id,
                    'score': score,
                    'dimension_scores': dimension_scores
                })
    
    def test_instructional_variations(self):
        """Test instruction adherence."""
        print("\n" + "=" * 60)
        print("TESTING: Instructional Variations (Instruction Adherence)")
        print("=" * 60)
        
        category_entries = [e for e in self.dataset['test_entries']
                           if e['category'] == 'instructional_variations']
        
        for entry in category_entries:
            if 'variations' not in entry:
                continue
            
            test_id = entry['test_id']
            base_topic = entry['base_query']
            
            print(f"\nTest {test_id} - Base: {base_topic}")
            
            variations = entry['variations']
            if len(variations) < 2:
                continue
            
            # Compare first two variations (should have different instructions)
            var1 = variations[0]
            var2 = variations[1]
            
            print(f"  Variation 1: {var1['query']} [{var1['instruction_type']}]")
            print(f"  Variation 2: {var2['query']} [{var2['instruction_type']}]")
            
            # Get embeddings
            emb1 = self.embedding_model.to_embeddings(var1['query'])
            emb2 = self.embedding_model.to_embeddings(var2['query'])
            
            # Evaluate
            src_dict = {'question': var2['query'], 'embedding': emb2}
            cache_dict = {
                'question': var1['query'],
                'answer': var1['ground_truth_answer'],
                'embedding': emb1
            }
            
            score = self.evaluator.evaluation(src_dict, cache_dict)
            dimension_scores = self.evaluator.get_dimension_scores(src_dict, cache_dict)
            
            # Should have lower instruction adherence score
            passed = dimension_scores['instruction_adherence'] < 0.9 or score < 0.9
            
            print(f"  Overall Score: {score:.3f}")
            print(f"  Instruction Score: {dimension_scores['instruction_adherence']:.3f}")
            print(f"  Result: {'✓ PASS (detected format difference)' if passed else '✗ FAIL (missed format difference)'}")
            
            if passed:
                self.results['instructional_variations']['passed'] += 1
            else:
                self.results['instructional_variations']['failed'] += 1
                self.results['instructional_variations']['details'].append({
                    'test_id': test_id,
                    'score': score,
                    'dimension_scores': dimension_scores
                })
    
    def test_true_negatives(self):
        """Test rejection of nonsensical queries."""
        print("\n" + "=" * 60)
        print("TESTING: True Negatives (Overall Quality)")
        print("=" * 60)
        
        category_entries = [e for e in self.dataset['test_entries']
                           if e['category'] == 'true_negatives']
        
        # Create a reasonable cached query to compare against
        reasonable_query = "How do machine learning algorithms work?"
        reasonable_emb = self.embedding_model.to_embeddings(reasonable_query)
        
        for entry in category_entries:
            test_id = entry['test_id']
            nonsense_query = entry['query']
            
            print(f"\nTest {test_id}:")
            print(f"  Nonsensical Query: {nonsense_query}")
            
            # Get embedding
            nonsense_emb = self.embedding_model.to_embeddings(nonsense_query)
            
            # Evaluate against reasonable query
            src_dict = {'question': nonsense_query, 'embedding': nonsense_emb}
            cache_dict = {
                'question': reasonable_query,
                'answer': "Machine learning algorithms learn patterns from data...",
                'embedding': reasonable_emb
            }
            
            score = self.evaluator.evaluation(src_dict, cache_dict)
            
            # Should have low score
            passed = score < 0.5
            
            print(f"  Overall Score: {score:.3f}")
            print(f"  Result: {'✓ PASS (correctly rejected)' if passed else '✗ FAIL (should reject)'}")
            
            if passed:
                self.results['true_negatives']['passed'] += 1
            else:
                self.results['true_negatives']['failed'] += 1
                self.results['true_negatives']['details'].append({
                    'test_id': test_id,
                    'score': score
                })
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.results.items():
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            
            total_passed += passed
            total_failed += failed
            
            if total > 0:
                success_rate = (passed / total) * 100
                status = "✓" if success_rate >= 70 else "✗"
                
                print(f"\n{status} {category}:")
                print(f"   Passed: {passed}/{total} ({success_rate:.1f}%)")
                
                if results['details']:
                    print(f"   Failed tests: {[d['test_id'] for d in results['details']]}")
        
        print(f"\n{'=' * 60}")
        overall_total = total_passed + total_failed
        overall_rate = (total_passed / overall_total * 100) if overall_total > 0 else 0
        
        print(f"OVERALL: {total_passed}/{overall_total} ({overall_rate:.1f}%)")
        
        if overall_rate >= 80:
            print("🎉 EXCELLENT - Advanced Correctness working well!")
        elif overall_rate >= 60:
            print("👍 GOOD - Acceptable performance, room for improvement")
        else:
            print("⚠️  NEEDS WORK - Consider tuning thresholds and weights")
        
        print("=" * 60)


def main():
    """Run validation tests."""
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        '../../evaluation_datasets/advanced_correctness_test_dataset.json'
    )
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the evaluation dataset exists.")
        return
    
    print("=" * 60)
    print("ADVANCED CORRECTNESS VALIDATION TEST")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    
    validator = AdvancedCorrectnessValidator(dataset_path)
    
    # Run tests
    validator.test_paraphrased_pairs()
    validator.test_factual_variations()
    validator.test_instructional_variations()
    validator.test_true_negatives()
    
    # Print summary
    validator.print_summary()


if __name__ == "__main__":
    main()


