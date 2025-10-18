import subprocess
import os

# --- 🧪 Define Your Experiment Grid ---
# This is where you configure all the experiments you want to run.

EMBEDDING_MODELS = [
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large",
]

# A more structured grid to handle sub-algorithms
SIMILARITY_ALGORITHMS = {
    "search_distance": {
        "metrics": {
            "ip": [0.85, 0.9, 0.95],      # Cosine Similarity (higher is better)
            "l2": [0.15, 0.2, 0.25, 0.3] # L2 Distance (lower is better)
        }
    },
    "sbert_crossencoder": {
        "metrics": {"none": [0.95]} # Metric is not applicable, but we need a placeholder
    },
    "sequence_match": {
        "metrics": {"none": [0.7, 0.8, 0.9]}
    },
    "exact_match": {
        "metrics": {"none": [1.0]}
    },
}

SEED = 42
OUTPUT_DIR = "results"
PYTHON_EXECUTABLE = "python" # Or "python3" if that's what you use

# -----------------------------------------

def run_experiment(model: str, algo: str, metric: str, threshold: float):
    """Constructs and runs a single experiment command."""
    print("---" * 20)
    print(f"🚀 STARTING EXPERIMENT:")
    print(f"  - Model: {model}")
    print(f"  - Algorithm: {algo}")
    if metric != "none":
        print(f"  - Vector Metric: {metric}")
    print(f"  - Threshold: {threshold}")
    print(f"  - Seed: {SEED}")
    print("---" * 20)

    command = [
        PYTHON_EXECUTABLE,
        "experiment.py",
        "--embedding-model", model,
        "--similarity-algo", algo,
        "--similarity-threshold", str(threshold),
        "--seed", str(SEED),
        "--output-dir", OUTPUT_DIR,
    ]

    # Add the vector-metric argument ONLY for the search_distance algorithm
    if algo == "search_distance" and metric != "none":
        command.extend(["--vector-metric", metric])

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ SUCCESS: {model} | {algo} ({metric}) | {threshold}")

    except subprocess.CalledProcessError as e:
        # This block is executed if the experiment script fails
        print(f"❌ FAILED: {model} | {algo} ({metric}) | {threshold}")
        print("--- STDOUT (from experiment.py) ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        print("--------------")
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find '{PYTHON_EXECUTABLE}'. Is your Conda environment active?")
        exit(1)


if __name__ == "__main__":
    # Calculate total number of runs
    total_runs = 0
    for model in EMBEDDING_MODELS:
        for algo, config in SIMILARITY_ALGORITHMS.items():
            for metric, thresholds in config["metrics"].items():
                total_runs += len(thresholds)
    current_run = 0

    print("🔔 Make sure your local Ollama server is running in the background.")
    print(f"Starting a total of {total_runs} experiments...")

    for model in EMBEDDING_MODELS:
        for algo, config in SIMILARITY_ALGORITHMS.items():
            for metric, thresholds in config["metrics"].items():
                for threshold in thresholds:
                    current_run += 1
                    # Construct expected summary path for skip logic
                    algo_path_name = f"{algo}-{metric}" if algo == "search_distance" else algo
                    summary_path = os.path.join(
                        OUTPUT_DIR,
                        model,
                        algo_path_name,
                        str(threshold),
                        f"seed_{SEED}",
                        "summary.csv",
                    )

                    if os.path.exists(summary_path):
                        print(f"\n\n>>> [{current_run}/{total_runs}] - ✅ SKIP >>>")
                        print(f"Results already exist for: {model} | {algo} ({metric}) | {threshold}")
                        continue

                    print(f"\n\n>>> [{current_run}/{total_runs}] - 🚀 RUN >>>")
                    run_experiment(model, algo, metric, threshold)

    print("\n\n🎉 All experiments complete!")