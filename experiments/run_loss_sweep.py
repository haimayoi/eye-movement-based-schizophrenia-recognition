"""
Loss Function Hyperparameter Sweep for BiCA-HS.
Runs 5 key configurations of Focal Loss parameters (alpha, gamma, lambda) and logs the results.

Usage:
    PYTHONPATH=. python experiments/run_loss_sweep.py --config configs/bica_config.yaml --seed 42
"""

import os
import argparse
import subprocess
import json
import pandas as pd

# Define configurations to test
SWEEP_CONFIGS = [
    {
        "name": "Cross Entropy (Baseline)",
        "alpha": 0.5,
        "gamma": 0.0,
        "lambda": 0.0
    },
    {
        "name": "Focal Loss Only (No Regularization)",
        "alpha": 0.5,
        "gamma": 2.0,
        "lambda": 0.0
    },
    {
        "name": "BiCA-HS Default (Focal + Entropy)",
        "alpha": 0.5,
        "gamma": 2.0,
        "lambda": 0.01
    },
    {
        "name": "High Focusing (Gamma = 3.0)",
        "alpha": 0.5,
        "gamma": 3.0,
        "lambda": 0.01
    },
    {
        "name": "High Regularization (Lambda = 0.05)",
        "alpha": 0.5,
        "gamma": 2.0,
        "lambda": 0.05
    }
]

def main():
    parser = argparse.ArgumentParser(description="Run Hyperparameter Sweep for Loss Function")
    parser.add_argument("--config", type=str, default="configs/bica_config.yaml", help="Path to base config file")
    parser.add_argument("--seed", type=int, default=42, help="Override random seed")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    results = []
    
    for i, cfg in enumerate(SWEEP_CONFIGS):
        print(f"\n================================================================================")
        print(f" Running Sweep {i+1}/{len(SWEEP_CONFIGS)}: {cfg['name']}")
        print(f" Params: alpha={cfg['alpha']}, gamma={cfg['gamma']}, lambda={cfg['lambda']}")
        print(f"================================================================================")
        
        # Temp output dir for this run
        out_dir = f"results/loss_sweep_cfg{i}_s{args.seed}"
        
        # Build command
        cmd = [
            "python", 
            "Bidirectional Cross-Attention Hybrid Stream/scripts/train_bica.py",
            "--config", args.config,
            "--seed", str(args.seed),
            "--batch-size", str(args.batch_size),
            "--output_dir", out_dir,
            "--focal-alpha", str(cfg["alpha"]),
            "--focal-gamma", str(cfg["gamma"]),
            "--entropy-lambda", str(cfg["lambda"])
        ]
        
        # Set PYTHONPATH to root directory to allow local package imports
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        # Run training
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during training config {cfg['name']}: {e}")
            continue
            
        # Read the generated summary
        summary_path = os.path.join(out_dir, "bica_results_summary.json")
        if not os.path.exists(summary_path):
            print(f"Error: results summary not found at {summary_path}")
            continue
            
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            
        overall = summary.get("overall", {})
        
        results.append({
            "Config Name": cfg["name"],
            "Alpha (a)": cfg["alpha"],
            "Gamma (g)": cfg["gamma"],
            "Lambda (l)": cfg["lambda"],
            "Accuracy": overall.get("accuracy", 0.0),
            "AUC": overall.get("auc", 0.0),
            "F1": overall.get("f1", 0.0),
            "Precision": overall.get("precision", 0.0),
            "Recall": overall.get("recall", 0.0),
            "Specificity": overall.get("specificity", 0.0)
        })
        
    if results:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        csv_path = "results/loss_sweep_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n=== Sweep Complete! Results saved to {csv_path} ===")
        
        # Print Table
        print(df.to_string(index=False))
    else:
        print("\nNo results gathered.")

if __name__ == "__main__":
    main()
