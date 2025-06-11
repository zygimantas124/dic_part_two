import sys
import logging

from util.helpers import parse_args, setup_logger
from util.training import train_ppo, train_dqn


# ---------- Entry Point ----------
if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(log_level=logging.DEBUG)

    logger.info(f"Training started with algorithm: {args.algo}")
    logger.info(f"Args: {args}")

    if args.algo == "ppo":
        train_ppo(args, logger)
    elif args.algo == "dqn":
        train_dqn(args, logger)
    else:
        logger.error(f"Unknown algorithm specified: {args.algo}")
        sys.exit(1)
