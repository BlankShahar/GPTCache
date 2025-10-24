#!/usr/bin/env python3
"""
prepare_oasst_workload.py

Prepares a realistic LLM caching workload from the OpenAssistant/oasst1 dataset.

Process:
1. Load oasst1 dataset from HuggingFace
2. Filter high-quality initial prompts (role='prompter', parent_id=null, rank>=3.0)
3. Generate embeddings using Ollama
4. Cluster prompts semantically using K-Means
5. Generate access patterns with Zipf distribution (realistic popularity)
6. Save workload to JSON

Usage:
    # On login node with internet (download dataset + generate embeddings)
    python prepare_oasst_workload.py --embedding-model nomic-embed-text --output workload_oasst.json
    
    # With custom parameters
    python prepare_oasst_workload.py \
        --embedding-model mxbai-embed-large \
        --num-clusters 10 \
        --num-prompts 800 \
        --num-queries 2000 \
        --min-rank 3.0 \
        --zipf-exponent 1.1 \
        --seed 42 \
        --output workload_oasst.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

# HuggingFace datasets
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not found. Install with: pip install datasets")
    sys.exit(1)

# OpenAI client (for Ollama's OpenAI-compatible API)
try:
    import openai
except ImportError:
    print("ERROR: 'openai' library not found. Install with: pip install openai")
    sys.exit(1)

# Scikit-learn for clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
except ImportError:
    print("ERROR: 'scikit-learn' library not found. Install with: pip install scikit-learn")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OASTWorkloadGenerator:
    """Generates realistic cache workload from OpenAssistant dataset."""
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://127.0.0.1:11434",
        num_clusters: int = 10,
        num_prompts: int = 800,
        num_queries: int = 2000,
        min_rank: float = 3.0,
        zipf_exponent: float = 1.1,
        seed: int = 42,
        language: str = "en"
    ):
        """
        Initialize workload generator.
        
        Args:
            embedding_model: Ollama model for embeddings (nomic-embed-text or mxbai-embed-large)
            ollama_host: Ollama server URL (default: http://127.0.0.1:11434)
            num_clusters: Number of semantic clusters
            num_prompts: Number of unique prompts to sample
            num_queries: Total queries in workload (includes repeats)
            min_rank: Minimum quality rank (0-5 scale, use >=3.0 for high quality)
            zipf_exponent: Zipf distribution exponent (higher = more skewed, 1.1 is realistic)
            seed: Random seed for reproducibility
            language: Language filter (default: 'en')
        """
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.num_clusters = num_clusters
        self.num_prompts = num_prompts
        self.num_queries = num_queries
        self.min_rank = min_rank
        self.zipf_exponent = zipf_exponent
        self.seed = seed
        self.language = language
        
        # Set random seeds
        np.random.seed(seed)
        
        # Setup OpenAI client for Ollama (same as experiment.py)
        # Handle both http://host:port and host:port formats
        if not ollama_host.startswith("http"):
            api_base = f"http://{ollama_host}/v1"
        else:
            api_base = ollama_host if ollama_host.endswith("/v1") else f"{ollama_host}/v1"
        
        openai.api_base = api_base
        openai.api_key = "ollama"  # Non-empty token required
        
        logger.info(f"Ollama API endpoint: {api_base}")
        
    def load_and_filter_dataset(self) -> List[Dict[str, Any]]:
        """
        Load oasst1 dataset and filter for high-quality initial prompts with ground truth answers.
        """
        logger.info("Loading OpenAssistant/oasst1 dataset from HuggingFace...")
        
        try:
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            logger.info(f"Loaded {len(dataset)} total messages")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.error("Make sure you're connected to the internet for first download")
            raise
        
        # --- EFFICIENT AND CORRECTED IMPLEMENTATION ---
        
        # Step 1: Create a map of parent_id -> list of child responses
        logger.info("Building a response map for efficient lookup...")
        responses_map: Dict[str, List[Dict[str, Any]]] = {}
        for msg in tqdm(dataset, desc="Mapping responses"):
            parent_id = msg.get('parent_id')
            if parent_id:
                if parent_id not in responses_map:
                    responses_map[parent_id] = []
                responses_map[parent_id].append(msg)

        # Step 2: Iterate through the dataset to find initial prompts that have high-quality answers
        logger.info(f"Filtering for prompts with answers ranked >= {self.min_rank} (lang='{self.language}')")
        
        filtered: List[Dict[str, Any]] = []
        no_response_count = 0
        
        for item in tqdm(dataset, desc="Filtering prompts"):
            # Condition 1: Is it an initial prompt from the right language?
            # LOGICAL ERROR FIX: We REMOVED the rank check on the initial prompt ('item').
            if (item['role'] == 'prompter' and 
                item['parent_id'] is None and 
                item['lang'] == self.language and
                not item.get('deleted', False)):
                
                prompt_id = item['message_id']
                
                # Condition 2: Does it have any responses in our map?
                if prompt_id in responses_map:
                    
                    # Condition 3: Filter for HIGH-QUALITY responses to this prompt.
                    high_quality_responses: List[Dict[str, Any]] = []
                    for resp in responses_map[prompt_id]:
                        resp_rank = resp.get('rank')
                        if (resp['role'] == 'assistant' and
                            # The rank check correctly belongs HERE, on the assistant's response.
                            (resp_rank is not None and resp_rank >= self.min_rank) and
                            not resp.get('deleted', False)):
                            
                            high_quality_responses.append({
                                'message_id': resp['message_id'],
                                'text': (resp.get('text') or '').strip(),
                                'rank': resp_rank
                            })
                    
                    # If we found at least one good answer, this is a valid pair.
                    if high_quality_responses:
                        # Sort by rank to find the absolute best one
                        high_quality_responses.sort(key=lambda x: x['rank'], reverse=True)
                        best_response = high_quality_responses[0]
                        
                        filtered.append({
                            'message_id': item['message_id'],
                            'text': (item.get('text') or '').strip(),
                            'rank': item.get('rank'), # Keep the original rank (usually None)
                            'lang': item['lang'],
                            'ground_truth': best_response['text'],
                            'ground_truth_id': best_response['message_id'],
                            'ground_truth_rank': best_response['rank'],
                            'num_responses': len(high_quality_responses)
                        })
                else:
                    no_response_count += 1
        
        # --- END OF CORRECTION ---
        
        logger.info(f"Filtered to {len(filtered)} high-quality prompts WITH ground truth answers")
        logger.info(f"Excluded {no_response_count} prompts that were un-answered or had low-rank answers")
        
        if len(filtered) == 0:
            raise ValueError("No prompts with ground truth found! Try lowering min_rank or checking language code")
        
        # Sample if we have more than needed
        if len(filtered) > self.num_prompts:
            logger.info(f"Sampling {self.num_prompts} prompts from {len(filtered)} candidates")
            # Sample with probability proportional to the rank of its BEST answer.
            answer_ranks = np.array([p['ground_truth_rank'] for p in filtered])
            probs = answer_ranks / answer_ranks.sum()
            indices = np.random.choice(len(filtered), size=self.num_prompts, replace=False, p=probs)
            filtered = [filtered[i] for i in indices]
        else:
            logger.warning(f"Only {len(filtered)} prompts available (requested {self.num_prompts})")
            self.num_prompts = len(filtered)
        
        return filtered
    
    def get_embeddings(self, prompts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for prompts using Ollama via OpenAI-compatible API.
        
        Args:
            prompts: List of prompt dictionaries
            
        Returns:
            Numpy array of shape (num_prompts, embedding_dim)
        """
        logger.info(f"Generating embeddings using {self.embedding_model}...")
        
        embeddings = []
        texts = [prompt['text'] for prompt in prompts]
        
        # Process in batches to avoid overwhelming the API
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding prompts"):
            batch = texts[i:i + batch_size]
            
            try:
                # Use OpenAI-compatible API (same as experiment.py)
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeddings = [item['embedding'] for item in response['data']]
                
                # Normalize embeddings (optional but recommended for cosine similarity)
                for emb in batch_embeddings:
                    norm = (sum(x * x for x in emb) ** 0.5) or 1.0
                    emb_normalized = [x / norm for x in emb]
                    embeddings.append(emb_normalized)
                    
            except Exception as e:
                logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
                # Use zero vectors as fallback for failed batches
                emb_dim = 768 if not embeddings else len(embeddings[0])
                for _ in range(len(batch)):
                    embeddings.append([0.0] * emb_dim)
        
        embeddings_array = np.array(embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def cluster_prompts(
        self, 
        prompts: List[Dict[str, Any]], 
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Cluster prompts using K-Means on embeddings.
        
        Args:
            prompts: List of prompt dictionaries
            embeddings: Embedding vectors
            
        Returns:
            Prompts with added 'cluster' field
        """
        logger.info(f"Clustering {len(prompts)} prompts into {self.num_clusters} semantic clusters...")
        
        # Normalize embeddings (cosine similarity = dot product of normalized vectors)
        embeddings_normalized = normalize(embeddings, norm='l2')
        
        # K-Means clustering
        kmeans = KMeans(
            n_clusters=min(self.num_clusters, len(prompts)),  # Can't have more clusters than points
            random_state=self.seed,
            n_init=10,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(embeddings_normalized)
        
        # Add cluster labels to prompts
        for prompt, cluster_id in zip(prompts, cluster_labels):
            prompt['cluster'] = int(cluster_id)
        
        # Log cluster distribution
        cluster_counts = Counter(cluster_labels)
        logger.info(f"Cluster distribution: {dict(cluster_counts)}")
        
        return prompts
    
    def generate_access_pattern(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate realistic access pattern using Zipf distribution.
        
        Zipf law: frequency ∝ 1 / rank^α
        - α = 1.0: classical Zipf (very skewed)
        - α = 1.1: more skewed (default, realistic for search queries)
        - α = 0.5: less skewed
        
        Args:
            prompts: List of prompts with cluster assignments
            
        Returns:
            Workload as list of prompt dictionaries (with repeats)
        """
        logger.info(f"Generating access pattern for {self.num_queries} queries (Zipf α={self.zipf_exponent})...")
        
        num_prompts = len(prompts)
        
        # Generate Zipf probabilities
        ranks = np.arange(1, num_prompts + 1)
        zipf_probs = 1.0 / (ranks ** self.zipf_exponent)
        zipf_probs /= zipf_probs.sum()
        
        # Sample queries according to Zipf distribution
        query_indices = np.random.choice(
            num_prompts,
            size=self.num_queries,
            replace=True,
            p=zipf_probs
        )
        
        workload = [prompts[i] for i in query_indices]
        
        # Log access statistics
        unique_accessed = len(set(query_indices))
        most_common = Counter(query_indices).most_common(5)
        logger.info(f"Access statistics:")
        logger.info(f"  - Unique prompts accessed: {unique_accessed}/{num_prompts} ({100*unique_accessed/num_prompts:.1f}%)")
        logger.info(f"  - Top 5 most frequent: {[(prompts[i]['message_id'][:8], count) for i, count in most_common]}")
        
        # Per-cluster access distribution
        cluster_access = Counter([prompts[i]['cluster'] for i in query_indices])
        logger.info(f"  - Cluster access counts: {dict(cluster_access)}")
        
        return workload
    
    def save_workload(self, workload: List[Dict[str, Any]], output_path: Path):
        """
        Save workload to JSON file.
        
        Args:
            workload: List of queries (prompts with repeats)
            output_path: Output file path
        """
        logger.info(f"Saving workload to {output_path}...")
        
        # Prepare output format (keep essential fields including ground truth)
        output_data = []
        for item in workload:
            output_data.append({
                'cluster': item['cluster'],
                'prompt': item['text'],
                'message_id': item['message_id'],
                'lang': item['lang'],
                'ground_truth': item['ground_truth'],
                'ground_truth_rank': item['ground_truth_rank']
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(output_data)} queries to {output_path}")
        
        # Save metadata
        metadata = {
            'num_queries': len(output_data),
            'num_unique_prompts': len(set(item['message_id'] for item in output_data)),
            'num_clusters': len(set(item['cluster'] for item in output_data)),
            'embedding_model': self.embedding_model,
            'zipf_exponent': self.zipf_exponent,
            'min_rank': self.min_rank,
            'seed': self.seed,
            'language': self.language,
            'has_ground_truth': True
        }
        
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        logger.info("✅ Ground truth answers included for accuracy measurement")
    
    def generate(self, output_path: Path):
        """
        Full pipeline: load → filter → embed → cluster → generate → save.
        
        Args:
            output_path: Output JSON file path
        """
        try:
            # Step 1: Load and filter dataset
            prompts = self.load_and_filter_dataset()
            
            # Step 2: Generate embeddings
            embeddings = self.get_embeddings(prompts)
            
            # Step 3: Cluster prompts
            prompts_clustered = self.cluster_prompts(prompts, embeddings)
            
            # Step 4: Generate access pattern
            workload = self.generate_access_pattern(prompts_clustered)
            
            # Step 5: Save workload
            self.save_workload(workload, output_path)
            
            logger.info("✅ Workload generation complete!")
            
        except Exception as e:
            logger.error(f"❌ Workload generation failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic LLM cache workload from OpenAssistant dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='nomic-embed-text',
        choices=['nomic-embed-text', 'mxbai-embed-large'],
        help='Ollama embedding model'
    )
    parser.add_argument(
        '--ollama-host',
        type=str,
        default='http://127.0.0.1:11434',
        help='Ollama server URL'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=800,
        help='Number of unique prompts to sample'
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=2000,
        help='Total queries in workload (includes repeats)'
    )
    parser.add_argument(
        '--min-rank',
        type=float,
        default=3.0,
        help='Minimum quality rank (0-5 scale, >=3.0 for high quality)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language filter (e.g., en, de, es)'
    )
    
    # Clustering configuration
    parser.add_argument(
        '--num-clusters',
        type=int,
        default=10,
        help='Number of semantic clusters'
    )
    
    # Access pattern configuration
    parser.add_argument(
        '--zipf-exponent',
        type=float,
        default=1.1,
        help='Zipf distribution exponent (1.1 is realistic for search queries)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='workload_oasst.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_queries < args.num_prompts:
        parser.error(f"--num-queries ({args.num_queries}) must be >= --num-prompts ({args.num_prompts})")
    
    if args.num_clusters > args.num_prompts:
        logger.warning(f"--num-clusters ({args.num_clusters}) > --num-prompts ({args.num_prompts}), adjusting...")
        args.num_clusters = args.num_prompts
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate workload
    generator = OASTWorkloadGenerator(
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
        num_clusters=args.num_clusters,
        num_prompts=args.num_prompts,
        num_queries=args.num_queries,
        min_rank=args.min_rank,
        zipf_exponent=args.zipf_exponent,
        seed=args.seed,
        language=args.language
    )
    
    generator.generate(output_path)


if __name__ == '__main__':
    main()