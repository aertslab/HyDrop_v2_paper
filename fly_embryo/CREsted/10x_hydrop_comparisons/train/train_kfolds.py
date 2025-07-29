import argparse

import anndata
import crested
import numpy as np
import tensorflow as tf


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tech", type=str, required=True)
    parser.add_argument("--species", type=str, required=True)
    parser.add_argument("--k_folds", type=int, default=11)  # 10 vals, 1 test

    return parser.parse_args()


def main(args):
    if args.species == "fly":
        genome_path = ""../../../../../../../../eceksi/resources/dmel/no_chr/dm6_nochr_filtered.fa""
        if args.tech == "hydrop":
            adata_path = "data/hydrop_fly.h5ad" # change to adata path in zip
        elif args.tech == "10x":
            adata_path = "data/10x_fly.h5ad" # change to adata path in zip
        else:
            raise ValueError("Unknown technology")
    # Preprocess
    adata = anndata.read_h5ad(adata_path)

    genome = crested.Genome(genome_path)
    crested.register_genome(genome)

    # Assign fold indices to regions
    num_regions = adata.n_vars
    fold_indices = np.arange(num_regions) % args.k_folds
    np.random.shuffle(fold_indices)

    adata.var["split"] = "train"
    adata.var["fold"] = fold_indices
    adata.var.loc[adata.var["fold"] == args.k_folds - 1, "split"] = (
        "test"  # Fixed test set
    )
    # save adata
    adata.write_h5ad(f"data/{args.tech}_{args.species}_kfolds.h5ad")

    # K-Fold training
    for fold in range(args.k_folds - 1):
        tf.keras.backend.clear_session()
        adata.var["split"] = "train"
        adata.var.loc[adata.var["fold"] == fold, "split"] = (
            "val"  # Rotate validation set
        )

        datamodule = crested.tl.data.AnnDataModule(
            adata,
            genome,
            batch_size=128,
            max_stochastic_shift=3,
            always_reverse_complement=True,
        )
        config_default = crested.tl.default_configs("peak_regression")
        config = crested.tl.TaskConfig(
            loss=crested.tl.losses.CosineMSELoss(), #default loss changed in crested
            metrics=config_default.metrics,
            optimizer=config_default.optimizer,
        )
        model_architecture = crested.tl.zoo.deeptopic_cnn(
            filters=512,
            conv_do=0.5,
            seq_len=500,
            num_classes=len(adata.obs_names),
            output_activation="softplus",
        )

        trainer = crested.tl.Crested(
            data=datamodule,
            config=config,
            model=model_architecture,
            project_name="Hydrop_paper_kfolds",  # Change to your liking
            run_name=f"{args.tech}_{args.species}_{fold}",  # Change to your liking
            logger="wandb",
        )

        trainer.fit()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
