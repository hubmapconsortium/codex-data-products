#!/usr/bin/env python3

import json
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from os import fspath, walk, listdir
from pathlib import Path
from typing import Dict, Tuple

import anndata
import muon as mu
import numpy as np
import os
import pandas as pd
import requests
import uuid
import yaml


def get_tissue_type(dataset: str) -> str:
    organ_dict = yaml.load(open("/opt/organ_types.yaml"), Loader=yaml.BaseLoader)
    organ_code = requests.get(
        f"https://entity.api.hubmapconsortium.org/dataset/{dataset}/organs/"
    )
    organ_name = organ_dict[organ_code]
    return organ_name.replace(" (Left)", "").replace(" (Right)", "")


def convert_tissue_code(tissue_code):
    with open("/opt/organ_types.yaml", 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    tissue_name = data.get(tissue_code)['description']
    return tissue_name


def find_files(directory, patterns):
    matched_files = []
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            for pattern in patterns:
                if filepath.match(pattern):
                    matched_files.append(filepath)
    return matched_files

def find_files_by_type(directory):
    hdf5_patterns = ["out.hdf5"]
    cell_count_patterns = ["reg1_stitched_expressions.ome.tiff-cell_channel_total.csv", "reg001_expr.ome.tiff-cell_channel_total.csv"]
    
    hdf5_files = find_files(directory, hdf5_patterns)
    cell_count_files = find_files(directory, cell_count_patterns)
    
    return hdf5_files, cell_count_files


def create_json(tissue, data_product_uuid, creation_time, uuids, hbmids, cell_count, file_size):
    bucket_url = f"https://hubmap-data-products.s3.amazonaws.com/{data_product_uuid}/"
    metadata = {
        "Data Product UUID": data_product_uuid,
        "Tissue": convert_tissue_code(tissue),
        "Assay": "atac",
        "Raw URL": bucket_url + f"{tissue}.h5mu",
        "Creation Time": creation_time,
        "Dataset UUIDs": uuids,
        "Dataset HBMIDs": hbmids,
        "Total Cell Count": cell_count,
        "Raw File Size": file_size
    }
    print("Writing metadata json")
    with open(f"{data_product_uuid}.json", "w") as outfile:
        json.dump(metadata, outfile)


def get_column_names(cell_count_file):
    cell_count_df = pd.read_csv(cell_count_file)
    column_names_list = list(cell_count_df.columns)
    column_names_list.remove('ID')
    return column_names_list


def create_anndata(hdf5_store, var_names, tissue_type, uuids_df):
    data_set_dir = fspath(hdf5_store.parent.stem)
    tissue_type = tissue_type if tissue_type else get_tissue_type(data_set_dir)
    store = pd.HDFStore(hdf5_store, 'r')
    key1 = '/total/channel/cell/expressions.ome.tiff/stitched/reg1'
    key2 = '/total/channel/cell/expr.ome.tiff/reg001'
    if key1 in store:
        matrix = store[key1]
    elif key2 in store:
        matrix = store[key2]
    store.close()
    # Create anndata object
    adata = anndata.AnnData(X=matrix, dtype=np.float64)
    adata.var_names = var_names
    adata.obs['ID'] = adata.obs.index
    adata.obs['dataset'] = data_set_dir
    cell_ids_list = [
        "-".join([data_set_dir, cell_id]) for cell_id in adata.obs["ID"]
    ]
    adata.obs["cell_id"] = pd.Series(
        cell_ids_list, index=adata.obs.index, dtype=str
    )
    adata.obs.set_index("cell_id", drop=True, inplace=True)
    return adata

def add_patient_metadata(obs, uuids_df):
    uuids_df["uuid"] = uuids_df["uuid"].astype(str)
    obs["dataset"] = obs["dataset"].astype(str)
    uuids_dict = uuids_df.set_index('uuid').to_dict(orient='index')
    for index, row in obs.iterrows():
        dataset_value = row['dataset']
        if dataset_value in uuids_dict:
            for key, value in uuids_dict[dataset_value].items():
                obs.at[index, key] = value
    del(obs["Unnamed: 0"])
    return obs


def main(data_dir, uuids_tsv, tissue):
    raw_output_file_name = f"{tissue}_raw.h5ad"
    uuids_df = pd.read_csv(uuids_tsv, sep="\t", dtype=str)
    uuids_list = uuids_df["uuid"].to_list()
    hbmids_list = uuids_df["hubmap_id"].to_list()    
    hdf5_files_list = []
    cell_count_files_list = []
    
    directories = [data_dir / Path(uuid) for uuid in uuids_df["uuid"]]
    
    for directory in directories:
        if len(listdir(directory)) > 1:
            hdf5_files, cell_count_files = find_files_by_type(directory)
            hdf5_files_list.extend(hdf5_files)
            cell_count_files_list.extend(cell_count_files)
    columns = get_column_names(cell_count_files[0])
    adatas = [create_anndata(file, columns, tissue, uuids_df) for file in hdf5_files_list]
    adata = anndata.concat(adatas, join="outer")
    obs_w_patient_info = add_patient_metadata(adata.obs, uuids_df)
    adata.obs = obs_w_patient_info
    creation_time = str(datetime.now())
    data_product_uuid = str(uuid.uuid4())
    total_cell_count = adata.obs.shape[0]
    adata.uns["creation_data_time"] = creation_time
    adata.uns["datasets"] = hbmids_list
    adata.uns["uuid"] = data_product_uuid
    print(adata.X)
    print(adata.X.dtype)
    adata.write(raw_output_file_name)
    file_size = os.path.getsize(raw_output_file_name)
    create_json(tissue, data_product_uuid, creation_time, uuids_list, hbmids_list, total_cell_count, file_size)    

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("data_directory", type=Path)
    p.add_argument("uuids_file", type=Path)
    p.add_argument("tissue", type=str, nargs="?")
    p.add_argument("--enable_manhole", action="store_true")

    args = p.parse_args()

    if args.enable_manhole:
        import manhole

        manhole.install(activate_on="USR1")

    main(args.data_directory, args.uuids_file, args.tissue)