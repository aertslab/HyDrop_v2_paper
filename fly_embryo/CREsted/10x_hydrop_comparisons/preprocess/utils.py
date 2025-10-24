import numpy as np
import wandb
import keras
import pysam
import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import LinearRegression


def get_hot_encoding_table(
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """Get hot encoding table to encode a DNA sequence to a numpy array with shape (len(sequence), len(alphabet)) using bytes."""

    def str_to_uint8(string) -> np.ndarray:
        """Convert string to byte representation."""
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    # 256 x 4
    hot_encoding_table = np.zeros(
        (np.iinfo(np.uint8).max + 1, len(alphabet)), dtype=dtype
    )

    # For each ASCII value of the nucleotides used in the alphabet
    # (upper and lower case), set 1 in the correct column.
    hot_encoding_table[str_to_uint8(alphabet.upper())] = np.eye(
        len(alphabet), dtype=dtype
    )
    hot_encoding_table[str_to_uint8(alphabet.lower())] = np.eye(
        len(alphabet), dtype=dtype
    )

    # For each ASCII value of the nucleotides used in the neutral alphabet
    # (upper and lower case), set neutral_value in the correct column.
    hot_encoding_table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    hot_encoding_table[str_to_uint8(neutral_alphabet.lower())] = neutral_value

    return hot_encoding_table


HOT_ENCODING_TABLE = get_hot_encoding_table()


def one_hot_encode_sequence(sequence: str, expand_dim: bool = True) -> np.ndarray:
    """One hot encode a DNA sequence."""
    if expand_dim:
        return np.expand_dims(
            HOT_ENCODING_TABLE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)],
            axis=0,
        )
    else:
        return HOT_ENCODING_TABLE[
            np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        ]


class GTvsPredCallback(keras.callbacks.Callback):
    def __init__(
        self, regions: list, y: np.array, cell_types: list, genome_path: str, **kwargs
    ):
        super().__init__(**kwargs)
        genome = pysam.FastaFile(genome_path)
        one_hots = []
        for region in regions:
            chromosome, start_end = region.split(":")
            start, end = start_end.split("-")
            start, end = int(start), int(end)
            sequence = genome.fetch(chromosome, start, end).upper()
            one_hot = one_hot_encode_sequence(sequence)
            one_hots.append(one_hot)
        self.regions = regions
        self.sequences = np.concatenate(one_hots, axis=0)
        self.ground_truth = y
        self.cell_types = cell_types

    def on_epoch_end(self, epoch, logs=None):
        # Run prediction on the sample input
        if self.model:  # Ensure the model is available
            predictions = self.model.predict(self.sequences)
            # Create a bar plot of prediction vs ground truth
            for i, (region, prediction) in enumerate(zip(self.regions, predictions)):
                # Create a bar plot of prediction vs ground truth for each region
                plt.figure(figsize=(10, 5))
                index = np.arange(len(prediction))  # Index for the bars
                bar_width = 0.35

                plt.bar(
                    index - bar_width / 2,
                    self.ground_truth[i],
                    bar_width,
                    label="Ground Truth",
                )
                plt.bar(
                    index + bar_width / 2, prediction, bar_width, label="Prediction"
                )

                plt.xlabel("Cell Types")
                plt.ylabel("Value")
                plt.title(
                    f"Predictions vs Ground Truth at Epoch {epoch} for Region {region}"
                )
                plt.legend()
                plt.xticks(index, self.cell_types, rotation=45)
                plt.tight_layout()

                # Save the plot to an in-memory file and log it to WandB
                wandb.log(
                    {
                        "epoch": epoch,
                        f"prediction_vs_gt_{region}": wandb.Image(
                            plt, caption=f"Epoch {epoch} - Region {region}"
                        ),
                    }
                )
                plt.close()


def normalize_accessibility_per_chromosome(
    adata: AnnData, target_mean: float = 1
) -> AnnData:
    """
    Normalizes the chromatin accessibility data in an AnnData object based on the
    average accessibility per chromosome, ensuring the mean accessibility is the same
    across all chromosomes.

    Parameters:
    - adata: AnnData object containing chromatin accessibility data.
    - target_mean: The desired target mean for normalization (default is 1).

    Returns:
    - AnnData: A new AnnData object with normalized chromatin accessibility values.
    """
    df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)

    chromosomes = adata.var["chr"]

    # Calculate the average accessibility per chromosome across all cell types
    average_accessibility_per_chromosome = (
        df.groupby(chromosomes, axis=1).mean().mean(axis=0)
    )

    # Compute scaling factors to normalize mean accessibility per chromosome
    scaling_factors = target_mean / average_accessibility_per_chromosome

    # Normalize the .X matrix by dividing by the scaling factor for each chromosome
    normalized_df = df.copy()

    for chromosome, regions in chromosomes.groupby(chromosomes):
        scaling_factor = scaling_factors[chromosome]
        normalized_df.loc[:, regions.index] *= scaling_factor
    normalized_adata = adata.copy()
    normalized_adata.X = normalized_df.values

    return normalized_adata


def normalize_peaks_for_gc_content(adata: AnnData, fasta_path: str) -> AnnData:
    """
    Normalize peak heights (accessibility) for GC content using linear regression.
    Sequences are fetched from the genome fasta file using pysam.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing peak heights in .X and region information in .var_names.
    fasta_path : str
        Path to the genome fasta file to fetch the sequences.

    Returns
    -------
    Adata with normalized peak heights (residuals from GC content regression).
    """

    def _gc_content(sequence):
        return (sequence.count("G") + sequence.count("C")) / len(sequence)

    fasta = pysam.FastaFile(fasta_path)
    all_sequences = []
    for region in adata.var_names:
        chrom, coords = region.split(":")
        start, end = coords.split("-")
        seq = fasta.fetch(chrom, int(start), int(end))
        all_sequences.append(seq)

    gc_contents = np.array([_gc_content(seq) for seq in all_sequences])

    normalized_peaks = np.zeros(adata.X.shape)  # Shape: (cell_types, regions)

    gc_contents = gc_contents.reshape(-1, 1)  # Reshape to (N_regions, 1) for regression

    for cell_type in range(adata.X.shape[0]):  # Iterate over each cell type
        peak_heights = adata.X[cell_type, :]  # Shape: (N_regions,)

        model = LinearRegression()
        model.fit(gc_contents, peak_heights)

        predicted_peak_heights = model.predict(gc_contents)

        # Calculate the residuals (actual - predicted) as the normalized peak heights
        residuals = peak_heights - predicted_peak_heights

        # Shift the residuals so that the minimum value is 0
        residuals_shifted = residuals - np.min(residuals)

        normalized_peaks[cell_type, :] = residuals_shifted

    adata_normalized = adata.copy()
    adata_normalized.X = normalized_peaks

    return adata_normalized
