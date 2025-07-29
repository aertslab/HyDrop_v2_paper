from anndata import AnnData


def split_by_chromosome_folds(adata: AnnData, k: int) -> None:
    """
    Split each chromosome into k equal folds and add 'fold_{i}' columns to adata.var.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with regions in `.var_names` and chromosome in the name.
    k : int
        Number of folds to split each chromosome into.

    Returns
    -------
    None (adds k columns to adata.var in-place: 'fold_0', 'fold_1', ..., 'fold_{k-1}')
    """
    if k < 2:
        raise ValueError("Number of folds `k` must be at least 2.")

    # Initialize fold columns with False
    for i in range(k):
        adata.var[f"fold_{i}"] = "train"

    # Split each chromosome separately
    for chrom, chrom_df in adata.var.groupby("chr"):
        indices = chrom_df.index.to_list()

        fold_sizes = [len(indices) // k] * k
        for i in range(len(indices) % k):
            fold_sizes[i] += 1

        start = 0
        for i, size in enumerate(fold_sizes):
            fold_indices = indices[start : start + size]
            adata.var.loc[fold_indices, f"fold_{i}"] = "val"
            start += size
