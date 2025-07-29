# 10x vs Hydropv2 k-fold comparisons

The analyses in this folder reproduce the paper's results on comparing the deep learning models trained on 10x and Hydropv2 data using the CREsted package.

The raw and preprocessed data can be found on in Zipfiles on Zenodo.
The only data file you need to provide yourself is a mm10 genome.fasta file.

## Preprocess

The preprocessing steps take the bigwigs from all experiment scenario's and convert them to the anndata format, while normalizing and filtering on specific peaks for finetuning. 
The resulting anndatas can also be found in the Zip folder.

## Train

The train scripts take the preprocessed anndata files and train 10-fold models. For each fold per experiment scenario, we train a baseline model on all peaks and select the best model based on the validation loss. That model is then finetuned further on the specific peak set. The best model per fold is saved, and these are the ones that are used for the analyses (these model files can be found in the Zenodo Zip folder).

To run a full k-fold training loop for one scenario, run the *run_k_fold.py* script.
```bash
# example 10x allcells allreads
python run_k_fold.py --adata_full_path "PREPROCESSED_FOLDER/10x_allcells_allreads/normalized.h5ad" --adata_specific_path "PREPROCESSED_FOLDER/10x_allcells_allreads/normalized_specific.h5ad" --genome_path "/path/to/mm10.fasta" --experiment_name "10x_allcells_allreads"
```

## Analyses

The analyses take the preprocessed anndatas and the finetuned models as input and generate the paper's figures. 

### correlation_barplots.ipynb

This notebook takes the finetuned models and compares the test set pearson correlation between the different scenarios.  
For all scenarios, the "all cells, all reads" scenario is taking as the test set (per technology), since we want to evaluate how well each scenario's model performs on the hold-out test set of the best possible quality to capture the "ground truth". If we were to validate on its own test set, our conclusions would be obscured by the lesser quality of the test set itself, instead of the quality of each model.  

### correlation_barplots_downsampled_cells.ipynb

This does the same as the above notebook, but instead of comparing the technologies on downsampled reads and downsampled cells, it evaluates how many cells are needed for Hydrop to achieve comparable performance.

### region_contributions.ipynb

This notebook plots interesting region contribution scores (as logo plots) and region predictions (as barplots) for the "all cells all reads" scenario over the 10-folds (average of predictions).  

### gene_locus_scoring.ipynb

This notebook uses the HyDropv2 "all cells, all reads" 10-fold models to score a gene locus around the Chys3 gene for L6 CT.

### tf_modisco_shared.ipynb

This notebook compares the HyDropv2 and 10x "all cells, all reads" models by detecting seqlets in the most specific regions per cell type and running tf-modisco.  
The resulting patterns are plotted as comparison clustermaps for 5 cell types of interest.  
If you only want to run the plotting functions; the tfmodisco results and pre-calculated contribution scores can be found in the Zip folder.

### validated_enhancers.ipynb

This notebook compares the HyDropv2 and 10x models by plotting heatmaps of the average 10-fold predicted cell types (based on max accessibility) for validated BICCN enhancers.  