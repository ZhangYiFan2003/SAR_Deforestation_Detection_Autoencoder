import logging
import random
import sys

import numpy as np
import torch

from config.parse_args import parse_arguments
from pipeline.datasets.data_loader import ProcessedForestDataLoader
from pipeline.models.autoencoder import AE
from pipeline.models.variational_autoencoder import VAE
from pipeline.experiments import create_run_context
from pipeline.test.test_pipeline import test_model
from pipeline.train.train_pipeline import train_model


LOGGER = logging.getLogger(__name__)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def main(argv=None):
    args = parse_arguments(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.seed, args.deterministic)

    run = create_run_context(args)
    args.run_id = run.run_id
    args.results_path = str(run.root)
    args.checkpoint_dir = str(run.checkpoints)
    args.log_dir = str(run.logs)
    args.best_checkpoint = str(run.best_checkpoint)

    wrapper_class = {"AE": AE, "VAE": VAE}[args.model]
    if args.use_optuna:
        return train_model(args, None, wrapper_class)

    data = ProcessedForestDataLoader(args)
    autoencoder = wrapper_class(args, data=data)

    try:
        if args.train:
            train_model(args, autoencoder, wrapper_class)
        if args.test:
            test_model(args, autoencoder, data)
    finally:
        autoencoder.writer.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("SAR workflow failed")
        sys.exit(1)
