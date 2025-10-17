import os
from glob import glob
import pandas as pd


def main():
    summary_files = glob("slurm_results/**/summary.csv", recursive=True)
    if not summary_files:
        print("No summary files found under 'slurm_results/'.")
        return

    all_summaries = []
    for fpath in summary_files:
        parts = fpath.split(os.sep)
        try:
            # Expect: slurm_results/<model>/<threshold>/seed_<seed>/summary.csv
            model = parts[1]
            threshold = float(parts[2])
            seed = int(parts[3].replace("seed_", ""))
        except Exception:
            print(f"Skipping unrecognized path structure: {fpath}")
            continue

        df = pd.read_csv(fpath)
        df["embedding_model"] = model
        df["threshold"] = threshold
        df["seed"] = seed
        all_summaries.append(df)

    if not all_summaries:
        print("No valid summaries parsed.")
        return

    final_df = pd.concat(all_summaries, ignore_index=True)
    out_path = "final_results_summary.csv"
    final_df.to_csv(out_path, index=False)

    print(f"✅ Master summary created at '{out_path}'")
    print(final_df)


if __name__ == "__main__":
    main()
