import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import json
import uuid
import numpy as np
from scipy import stats
from datetime import datetime

from training import train
from evaluate import evaluate_agent
from helpers import parse_args, set_global_seed, setup_logger


def summarize_metric(metric_list, confidence=0.95):
    arr = np.array(metric_list)
    mean = arr.mean()
    std = arr.std()
    n = len(arr)
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return {"mean": mean, "ci": h, "n": n, "std": std}


def run_multiple_trainings(args, n_runs=20):
    logger = setup_logger()
    hausdorff_scores = []
    tortuosity_scores = []

    for i in range(n_runs):
        seed = i + 1
        set_global_seed(seed)
        run_id = str(uuid.uuid4())[:8]
        model_path = f"logs/temp_model_{run_id}.pth"

        args.seed = seed
        args.save_model_path = model_path
        args.evaluate_after_training = False
        args.load_model_path = None

        logger.info(f"\n========== Run {i + 1}/{n_runs} (Seed {seed}) ==========")
        train(args, logger)

        args.model_path = model_path
        args.eval_epsilon = 0.05
        args.n_episodes = 10

        metrics = evaluate_agent(args)
        if metrics:
            hausdorff_scores.append(metrics["avg_hausdorff"])
            tortuosity_scores.append(metrics["avg_tortuosity"])
        else:
            logger.warning(f"Run {i + 1} failed evaluation.")

        if os.path.exists(model_path):
            os.remove(model_path)

    summary = {
        "hausdorff": summarize_metric(hausdorff_scores),
        "tortuosity": summarize_metric(tortuosity_scores),
    }

    raw = {
        "hausdorff": hausdorff_scores,
        "tortuosity": tortuosity_scores,
    }

    return summary, raw


def save_results_to_json(results, args, all_scores, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Use config filename (e.g., DQN_walls_no_obst_no_rayc.txt) if provided
    config_name = None
    if hasattr(args, "config_file") and args.config_file:
        config_name = os.path.basename(args.config_file)  # e.g., "DQN_walls_no_obst_no_rayc.txt"
        setup_name = config_name.replace(".txt", "")  # e.g., "DQN_walls_no_obst_no_rayc"
    else:
        setup_name = f"{args.algo}_{args.env_name}"

    filename = f"{setup_name}_summary.json"
    filepath = os.path.join(output_dir, filename)

    results_to_save = {
        "setup_name": setup_name,
        "env_name": args.env_name,
        "algo": args.algo,
        "n_runs": results["hausdorff"]["n"],
        "args": vars(args),
        "individual_scores": {
            "hausdorff": all_scores["hausdorff"],
            "tortuosity": all_scores["tortuosity"],
        },
        "summary": {
            "hausdorff": results["hausdorff"],
            "tortuosity": results["tortuosity"],
        },
    }

    with open(filepath, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✅ Results saved to: {filepath}")


if __name__ == "__main__":
    args = parse_args()

    summary, raw_scores = run_multiple_trainings(args, n_runs=20)

    print("\n========== FINAL CONFIDENCE INTERVALS ==========")
    for k, v in summary.items():
        print(f"{k.capitalize():<12}: {v['mean']:.4f} ± {v['ci']:.4f} (95% CI over {v['n']} runs)")

    save_results_to_json(summary, args, raw_scores)
