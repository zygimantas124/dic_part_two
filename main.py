import sys
import logging
import shutil
import os


from util.helpers import parse_args, setup_logger
from util.training import train  

if __name__ == "__main__":
    log_dir = "./logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        
    args = parse_args()
    logger = setup_logger(log_level=logging.DEBUG)

    logger.info(f"Training started with algorithm: {args.algo.upper()}")
    logger.info(f"Args: {args}")

    if args.algo in ["ppo", "dqn"]:
        train(args, logger)
    else:
        logger.error(f"Unknown algorithm specified: {args.algo}")
        sys.exit(1)