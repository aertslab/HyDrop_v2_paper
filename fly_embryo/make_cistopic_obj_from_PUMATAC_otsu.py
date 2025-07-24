#__Modules__
print('Loading modules........')
import pandas as pd
import polars as pl
import pickle
import os
import re
import glob
import sys
import itertools
import scrublet as scr
import matplotlib.pyplot as plt
import plotnine as pn
from plotnine import *
from pycisTopic.cistopic_class import *
import pyranges
import shutil
print('all modules imported correctly')
print()
print()


#__Paths__
print('set up paths.......')
proj_dir = ""
outdir = "/staging/leuven/stg_00002/lcb/hydrop_v2_paper/fly/cistopic_LCB_embryo_consensuspeaks/HyDrop_10x_otsu_notdownsampled"
tmp_dir = '/scratch/leuven/350/vsc35050/HyDrop_v2/'
#metadata_path = proj_dir +  "/doc/samples_metadata_with_donor_region_info_250523.csv"
#qc_stats_path = "/lustre1/project/stg_00090/ASA/analysis/20230531_ATAC_QC_2/samples_combined_stats.csv"
consensus_peaks_path = '/staging/leuven/stg_00002/lcb/hdickm/resources/consensus_peaks/dev_consensus_peaks/consensus_regions_LCB_dechor_embryo_atlas_v1_082024.bed'
path_to_blacklist ='/staging/leuven/stg_00002/lcb/eceksi/resources/dm6-blacklist-nochr.v2.bed'
cistopic_objects_out = outdir + '/cistopic_objs/'
#annot = pd.read_csv('/staging/leuven/stg_00090/ASA/analysis/analysis_jdeman/tools/annot.csv', index_col = 0)
print('project dir: ' + proj_dir)
print('out dir: ' + outdir)
print('tmp dir: ' + tmp_dir)
#print('metadata: ' + metadata_path)
#print('qc: ' + qc_stats_path)
print('consensus peaks: ' + consensus_peaks_path)
print('cto dir: ' + cistopic_objects_out)
print('blacklist: ' + path_to_blacklist)
print('does blacklist exist? ' + str(os.path.exists(path_to_blacklist)))
print('all paths set up correctly')
print()
print()


#__set up dictionaries__
print('___DICTS___')
print('fragments dict')
fragments_dict_subset = {
    os.path.basename(x).split(".")[0]: x
    for x in sorted(
        sorted(
            glob.glob(            "/lustre1/project/stg_00002/lcb/fderop/data/20231115_hydrop_v2/analysis_drosophila/PUMATAC_out/data/fragments/*.fragments.tsv.gz"
            )
        )
    )
}
print(fragments_dict_subset)
print()
#print('fragments sub dict')
#fragments_dict_subset = dict((k, fragments_dict[k]) for k in metadata['short_bc_id'])
#print(fragments_dict_subset)
print()
print('consensus dict')
consensus_dict = {}
for k in fragments_dict_subset:
    consensus_dict[k] = consensus_peaks_path
print(consensus_dict)
print()
scale_samples_list = []

#__Check which CTOs exist__
print('___CHECKING EXISTING CTOs___')
fragments_sub_dict={}
for sample in fragments_dict_subset:
    cto = os.path.join(cistopic_objects_out, sample + "__cto.pkl")
    print(f"Checking if {cto} exist...")
    if os.path.exists(cto):
        print(f"\t{cto} exists! Skipping...")
    else:
        print(f"\t{cto} does not exist, adding to subdict to generate")
        fragments_sub_dict[sample] = fragments_dict_subset[sample]
        metadata_bc_sub_dict = {}
        bc_passing_filters_sub_dict = {}
        for sample in fragments_sub_dict.keys():
#            metadata_bc_sub_dict[sample] = f"/staging/leuven/stg_00090/ASA/analysis/analysis_Olga/0_cingulate_cortex_trials/20230531_ATAC_QC_2/cistopic_qc_out/{sample}__metadata_bc.pkl"
#            bc_passing_filters_sub_dict[
#                sample
#            ] = f"/staging/leuven/stg_00090/ASA/analysis/analysis_Olga/0_cingulate_cortex_trials/20230531_ATAC_QC_2/selected_barcodes/{sample}_bc_passing_filters.pkl"
            metadata_bc_sub_dict[sample] = f"/lustre1/project/stg_00002/lcb/fderop/data/20231115_hydrop_v2/analysis_drosophila/cistopic_qc_out/{sample}__metadata_bc.pkl"
            bc_passing_filters_sub_dict[sample] = f"/lustre1/project/stg_00002/lcb/fderop/data/20231115_hydrop_v2/analysis_drosophila/selected_barcodes/{sample}_bc_passing_filters_otsu.pkl"
print()
print('samples: ')
fragments_sub_dict.keys()

#__set output dir___
out = cistopic_objects_out
print('cistopic object output dir: ' + out)
print()
print()

#__make ctos___
print('MAKING CTOs.......')
if fragments_sub_dict != {}:
    from pycisTopic.cistopic_class import create_cistopic_object_from_fragments
    n_cores = 1
    for sample in fragments_sub_dict.keys():
        print(sample)
        cto_path = os.path.join(out, f"{sample}__cto.pkl")
        print(cto_path)
        if not sample in scale_samples_list:
            print(sample + ' not in scale_samples_list')
            if not os.path.exists(cto_path):
                print('iterating over tmp dir....')
                for files in os.listdir(tmp_dir):
                    path = os.path.join(tmp_dir, files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)
                
                print('reading metadata_bc for ' + sample + '....')
                print(metadata_bc_sub_dict[sample])
                infile = open(metadata_bc_sub_dict[sample], 'rb')
                metadata_bc = pickle.load(infile)
                metadata_bc.index = [bc.split("___")[0] for bc in metadata_bc.index]
                infile.close()
                print('reading bc_passing_filters for ' + sample + '....')
                print(bc_passing_filters_sub_dict[sample])
                infile = open(bc_passing_filters_sub_dict[sample], 'rb')
                bc_passing_filters = pickle.load(infile)
                infile.close()
                bc_passing_filters_fixed = [bc.split("___")[0] for bc in bc_passing_filters]
                print('start creating cto for ' + sample)
                cto = create_cistopic_object_from_fragments(path_to_fragments=fragments_sub_dict[sample],
                                                                path_to_regions=consensus_peaks_path,
                                                                path_to_blacklist=path_to_blacklist,
                                                                metrics=metadata_bc,
                                                                valid_bc=bc_passing_filters_fixed,
                                                                n_cpu=n_cores,
                                                                partition=10,
                                                                project=sample)
                print('creating cto for ' + sample + ' finished')
                cto_path = os.path.join(out, f"{sample}__cto.pkl")
                print(f"Writing {sample} cto in {cto_path}...")

                with open(
                    cto_path, "wb"
                ) as f:
                    pickle.dump(cto, f, protocol=4)

            else:
                print(f"{cto_path} already exists!")

    else:
        print("All samples already processed.")
        
print('___### SUCCESS ###___')
