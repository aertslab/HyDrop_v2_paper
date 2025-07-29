"""Use train function from train.py to run k-fold cross-validation."""

import argparse
import anndata
from loguru import logger

from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with Crested")
    parser.add_argument(
        "--adata_full_path",
        type=str,
        required=True,
        help="Path to the preprocessed AnnData file",
    )
    parser.add_argument(
        "--adata_specific_path",
        type=str,
        required=True,
        help="Path to the preprocessed AnnData file containing specific peaks only",
    )
    parser.add_argument(
        "--genome_path",
        type=str,
        required=True,
        help="Path to the fasta",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment for logging and checkpointing purposes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of epochs to train the model",
    )
    return parser.parse_args()


def main(
    args,
):
    adata_full = anndata.read_h5ad(args.adata_full_path)
    adata_specific = anndata.read_h5ad(args.adata_specific_path)

    # get n folds from the data
    n_folds = len([col for col in adata_full.var.columns if col.startswith("fold_")])
    logger.info(f"Number of folds: {n_folds}")

    for fold in range(n_folds):
        logger.info(f"Training fold {fold + 1}/{n_folds}")
        # change fold columns name to 'split' (required for crested)
        adata_full.var["split"] = adata_full.var[f"fold_{fold}"]
        adata_specific.var["split"] = adata_specific.var[f"fold_{fold}"]

        # train the model
        train(
            adata_full=adata_full,
            adata_specific=adata_specific,
            genome_path=args.genome_path,
            experiment_name=args.experiment_name,
            run_name=f"fold_{fold}",
            epochs=args.epochs,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
