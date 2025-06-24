import sys
import logging
import shutil
import os


from util.helpers import parse_args, setup_logger
from util.training import train, evaluate

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(log_level=logging.DEBUG)

    if args.evaluate_only:
        logger.info("Running evaluation only")
        if not evaluate(args, logger):
            sys.exit(1)
    else:
        log_dir = "./logs"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            
        logger.info(f"Training started with algorithm: {args.algo.upper()}")
        logger.info(f"Args: {args}")

        if args.algo in ["ppo", "dqn"]:
            train(args, logger)  # Evakuation handled in train
        else:
            logger.error(f"Unknown algorithm specified: {args.algo}")
            sys.exit(1)


# Example usage: TODO: someone checks if these are reaonsable

# 1. Training only (basic)
# python main.py --algo dqn --max_episodes 1000 --env_name open_office_simple --device cuda

# 2. Evaluation only (load existing model)
# python main.py --evaluate_only --load_model_path logs/my_model.pth --eval_episodes 20 --device cuda

# 3. Train then evaluate (complete pipeline - with and without saving)
# python main.py --algo ppo --max_episodes 80 --evaluate_after_training --eval_episodes 10 --device cuda
# python main.py --algo ppo --max_episodes 80 --evaluate_after_training --eval_episodes 10 --save_model_path logs/my_model --device cuda
