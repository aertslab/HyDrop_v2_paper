"""Train a single model (baseline + finetune) with Crested."""

import argparse
import anndata
import os
import glob

import crested
import keras


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
        "--run_name",
        type=str,
        required=True,
        help="Name of the run for logging and checkpointing purposes",
    )
    return parser.parse_args()


def _get_last_checkpoint(folder_path):
    checkpoint_files = glob.glob(os.path.join(folder_path, "*.keras"))
    checkpoint_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return checkpoint_files[-1]


def train(
    adata_full: anndata.AnnData,
    adata_specific: anndata.AnnData,
    genome_path: str,
    experiment_name: str,
    run_name: str,
    epochs: int = 60,
    learning_rate_reduce_patience: int = 3,
    early_stopping_patience: int = 6,
):
    # register genome
    genome = crested.Genome(genome_path, name="mm10")
    crested.register_genome(genome)

    # load data containing all regions
    datamodule = crested.tl.data.AnnDataModule(
        adata_full,
        batch_size=512,
        max_stochastic_shift=3,
        always_reverse_complement=True,
    )

    # Load chrombpnet-like architecture for a dataset with 2114bp wide regions
    model_architecture = crested.tl.zoo.dilated_cnn(
        seq_len=2114, num_classes=len(list(adata_full.obs_names))
    )

    # Load the default configuration for training a peak regression model
    config = crested.tl.default_configs("peak_regression")
    print(config)

    # setup the trainer & train
    trainer = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=config,
        project_name=f"hydrop_v2_review_mouse_{experiment_name}",
        run_name=f"{run_name}_baseline",
        logger="wandb",
        seed=7,
    )
    trainer.fit(
        epochs=epochs,
        learning_rate_reduce_patience=learning_rate_reduce_patience,
        early_stopping_patience=early_stopping_patience,
        save_dir=f"checkpoints/{experiment_name}/{run_name}_baseline",
    )

    # finetuning stage now
    del adata_full
    datamodule = crested.tl.data.AnnDataModule(
        adata_specific,
        batch_size=128,
        max_stochastic_shift=3,
        always_reverse_complement=True,
    )

    best_checkpoint_path = _get_last_checkpoint(
        f"checkpoints/{experiment_name}/{run_name}_baseline/checkpoints"
    )  # we're only saving the checkpoints if the val_loss improves
    print(f"Loading best checkpoint from {best_checkpoint_path}")
    model_architecture = keras.models.load_model(
        best_checkpoint_path,
        compile=False,
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # lower LR
    loss = config.loss
    metrics = config.metrics
    alternative_config = crested.tl.TaskConfig(optimizer, loss, metrics)
    print(alternative_config)

    trainer = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=alternative_config,
        project_name=f"hydrop_v2_review_mouse_{experiment_name}",
        run_name=f"{run_name}_finetune",
        logger="wandb",
        seed=7,
    )
    trainer.fit(
        epochs=epochs,
        learning_rate_reduce_patience=learning_rate_reduce_patience,
        early_stopping_patience=early_stopping_patience,
        save_dir=f"checkpoints/{experiment_name}/{run_name}_finetune",
    )


if __name__ == "__main__":
    args = parse_args()
    adata_full = anndata.read_h5ad(args.adata_full_path)
    adata_specific = anndata.read_h5ad(args.adata_specific_path)
    train(
        adata_full=adata_full,
        adata_specific=adata_specific,
        genome_path=args.genome_path,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        epochs=60,
        learning_rate_reduce_patience=3,
        early_stopping_patience=6,
    )
