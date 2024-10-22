#!/usr/bin/env python3

import json
import logging
import os
import re
import uuid
from argparse import ArgumentParser
from datetime import datetime
from os import fspath, listdir, walk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import requests
import yaml
from scipy.io import mmread
from scipy.sparse import block_diag

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)

antibodies_dict = {
    "BCL-2": "BCL2",
    "Collagen IV": ["CollIV", "CollagenIV", "collagen IV", "COLIV"],
    "Cytokeratin": "cytokeratin",
    "eCAD": ["E-CAD", "ECAD"],
    "HLA-DR": "HLADR",
    "Hoechst1": "HOECHST1",
    "PanCK": "panCK",
    "Podoplanin": ["Podoplan", "podoplanin", "PDPN"],
    "Synaptophysin": ["Synapt", "Synapto"],
    "aDefensin 5": ["aDef5", "aDefensin5"],
    "MUC-1/EMA": "MUC1",
    "NKG2D (CD314)": ["NKG2D", "NKG2G"],
    "a-SMA": ["SMActin", "aSMA", "SMA"],
    "MUC-2": "MUC2",
    "Foxp3": "FoxP3",
}


def find_antibodies_meta(input_dir: Path) -> Optional[Path]:
    """
    Looks for metadata files matching the pattern for antibodies in the given UUID directory.
    """
    metadata_filename_pattern = re.compile(r".*antibodies\.tsv$")
    found_files = []

    for filename in listdir(input_dir):
        if metadata_filename_pattern.match(filename):
            found_files.append(Path(input_dir) / filename)

    if found_files:
        return found_files[0]  # Return the first matching file
    else:
        logger.warning(f"No antibody file found in {input_dir}")
        return None


def get_analyte_name(antibody_name: str) -> str:
    """
    Strips unnecessary prefixes and suffixes off of antibody name from antibodies.tsv.
    """
    antb = re.sub(r"Anti-", "", antibody_name)
    antb = re.sub(r"\s+antibody", "", antb)
    antb = re.sub(r"antibody", "", antb)

    return antb


def find_antibody_key(value: str) -> str:
    value_lower = value.strip().lower()
    for key, val in antibodies_dict.items():
        if isinstance(val, str) and val.strip().lower() == value_lower:
            return key
        elif isinstance(val, list) and value_lower in [v.strip().lower() for v in val]:
            return key
    return value


def get_tissue_type(dataset: str) -> str:
    organ_dict = yaml.load(open("/opt/organ_types.yaml"), Loader=yaml.BaseLoader)
    organ_code = requests.get(
        f"https://entity.api.hubmapconsortium.org/dataset/{dataset}/organs/"
    )
    organ_name = organ_dict[organ_code]
    return organ_name.replace(" (Left)", "").replace(" (Right)", "")


def convert_tissue_code(tissue_code: str) -> str:
    with open("/opt/organ_types.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    tissue_name = data.get(tissue_code)["description"]
    return tissue_name


def find_files(directory: Path, patterns: list) -> list:
    matched_files = []
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            for pattern in patterns:
                if filepath.match(pattern):
                    matched_files.append(filepath)
    return matched_files


def find_files_by_type(directory: Path) -> Tuple:
    hdf5_patterns = ["out.hdf5"]
    cell_count_patterns = [
        "reg1_stitched_expressions.ome.tiff-cell_channel_total.csv",
        "reg001_expr.ome.tiff-cell_channel_total.csv",
    ]
    adjacency_matrix_patterns = [
        "reg1_stitched_expressions.ome.tiff_AdjacencyMatrix.mtx",
        "reg001_expr.ome.tiff_AdjacencyMatrix.mtx",
    ]
    adjacency_matrix_labels_patterns = [
        "reg1_stitched_expressions.ome.tiff_AdjacencyMatrixRowColLabels.txt",
        "reg001_expr.ome.tiff_AdjacencyMatrixRowColLabels.txt",
    ]
    cell_centers_patterns = [
        "reg1_stitched_expressions.ome.tiff-cell_centers.csv",
        "reg001_expr.ome.tiff-cell_centers.csv",
    ]

    hdf5_files = find_files(directory, hdf5_patterns)
    cell_count_files = find_files(directory, cell_count_patterns)
    adjacency_matrix_files = find_files(directory, adjacency_matrix_patterns)
    adjacency_matrix_labels_files = find_files(
        directory, adjacency_matrix_labels_patterns
    )
    cell_centers_files = find_files(directory, cell_centers_patterns)

    return (
        hdf5_files,
        cell_count_files,
        adjacency_matrix_files,
        adjacency_matrix_labels_files,
        cell_centers_files,
    )


def create_json(
    tissue: str,
    data_product_uuid: str,
    creation_time: str,
    uuids: list,
    hbmids: list,
    cell_count: int,
    file_size: int,
):
    bucket_url = f"https://hubmap-data-products.s3.amazonaws.com/{data_product_uuid}/"
    metadata = {
        "Data Product UUID": data_product_uuid,
        "Tissue": convert_tissue_code(tissue),
        "Assay": "CODEX",
        "Raw URL": bucket_url + f"{tissue}.h5mu",
        "Creation Time": creation_time,
        "Dataset UUIDs": uuids,
        "Dataset HBMIDs": hbmids,
        "Total Cell Count": cell_count,
        "Raw File Size": file_size,
    }
    print("Writing metadata json")
    with open(f"{data_product_uuid}.json", "w") as outfile:
        json.dump(metadata, outfile)


def get_column_names(cell_count_file: Path) -> list:
    cell_count_df = pd.read_csv(cell_count_file)
    column_names_list = list(cell_count_df.columns)
    column_names_list.remove("ID")
    return column_names_list


def standardize_antb_df(antibodies_df: pd.DataFrame) -> pd.DataFrame:
    for idx, row in antibodies_df.iterrows():
        stripped_name = get_analyte_name(antibodies_df.at[idx, "antibody_name"])
        new_name = find_antibody_key(stripped_name)
        antibodies_df.at[idx, "antibody_name"] = new_name
    return antibodies_df


def create_varm_dfs(
    adata: anndata.AnnData,
    uuid: str,
    antibodies_df: pd.DataFrame,
    var_antb_tsv_intersection: list,
):
    # Create an empty DataFrame for each piece of information with proteins as rows and dataset UUIDs as columns
    uniprot_df = pd.DataFrame(index=adata.var.index, columns=[uuid])
    rrid_df = pd.DataFrame(index=adata.var.index, columns=[uuid])
    antibodies_tsv_id_df = pd.DataFrame(index=adata.var.index, columns=[uuid])

    # Fill in the DataFrames with matching values from antibodies_df
    matching_antibodies = antibodies_df[
        antibodies_df["antibody_name"].isin(var_antb_tsv_intersection)
    ]
    for antibody in var_antb_tsv_intersection:
        protein_idx = adata.var.index.get_loc(antibody)
        uniprot_df.iloc[protein_idx, 0] = matching_antibodies.loc[
            matching_antibodies["antibody_name"] == antibody, "uniprot_accession_number"
        ].values[0]
        rrid_df.iloc[protein_idx, 0] = matching_antibodies.loc[
            matching_antibodies["antibody_name"] == antibody, "rr_id"
        ].values[0]
        antibodies_tsv_id_df.iloc[protein_idx, 0] = matching_antibodies.loc[
            matching_antibodies["antibody_name"] == antibody, "channel_id"
        ].values[0]
    return uniprot_df, rrid_df, antibodies_tsv_id_df


def create_anndata(
    hdf5_store: Path,
    tissue_type: str,
    uuids_df: pd.DataFrame,
    cell_centers_file: Path,
    cell_count_file: Path,
    data_directory: Path,
) -> anndata.AnnData:
    data_set_dir = fspath(hdf5_store.parent.stem)
    parent_uuid = uuids_df.loc[
        uuids_df["uuid"] == data_set_dir, "immediate_ancestor_ids"
    ].item()
    raw_dir = data_directory / parent_uuid
    antibodies_tsv = find_antibodies_meta(raw_dir)
    tissue_type = tissue_type if tissue_type else get_tissue_type(data_set_dir)
    store = pd.HDFStore(hdf5_store, "r")
    key1 = "/total/channel/cell/expressions.ome.tiff/stitched/reg1"
    key2 = "/total/channel/cell/expr.ome.tiff/reg001"
    var_names = get_column_names(cell_count_file)
    # Replace column names with antibody names using the dictionary
    var_names = [find_antibody_key(var) for var in var_names]    
    if antibodies_tsv:
        antibodies_df = pd.read_csv(antibodies_tsv, sep="\t", dtype=str)
        antibodies_df = standardize_antb_df(antibodies_df)
        antibodies_tsv_list = antibodies_df["antibody_name"].to_list()
        var_antb_tsv_intersection = [
            value for value in var_names if value in antibodies_tsv_list
        ]

    if key1 in store:
        matrix = store[key1]
        mean_layer_matrix = store[
            "/meanAll/channel/cell/expressions.ome.tiff/stitched/reg1"
        ]
    elif key2 in store:
        matrix = store[key2]
        mean_layer_matrix = store["/meanAll/channel/cell/expr.ome.tiff/reg001"]
    store.close()

    adata = anndata.AnnData(X=matrix, dtype=np.float64)
    adata.var_names = var_names
    adata.obs["ID"] = adata.obs.index
    adata.obs["dataset"] = data_set_dir

    # Set index for cell IDs
    cell_ids_list = ["-".join([data_set_dir, cell_id]) for cell_id in adata.obs["ID"]]
    adata.obs["cell_id"] = pd.Series(cell_ids_list, index=adata.obs.index, dtype=str)
    adata.obs.set_index("cell_id", drop=True, inplace=True)

    # Add the mean layer
    adata.layers["mean_expression"] = mean_layer_matrix

    # Read cell centers
    cell_centers_df = pd.read_csv(cell_centers_file)

    # Create the cell centers matrix and store it in .obsm
    adata.obsm["centers"] = cell_centers_df.loc[
        cell_centers_df["ID"].astype(str).isin(adata.obs["ID"].astype(str)), ["x", "y"]
    ].to_numpy()

    if antibodies_tsv and var_antb_tsv_intersection:
        uniprot_df, rrid_df, antb_tsv_id_df = create_varm_dfs(
            adata, data_set_dir, antibodies_df, var_antb_tsv_intersection
        )
        # Store these DataFrames in .varm with the dataset UUID as columns
        adata.varm["Uniprot_ID"] = uniprot_df
        adata.varm["RRID"] = rrid_df
        adata.varm["AntibodiesTsvID"] = antb_tsv_id_df

    return adata


def add_patient_metadata(obs: pd.DataFrame, uuids_df: pd.DataFrame) -> pd.DataFrame:
    uuids_df["uuid"] = uuids_df["uuid"].astype(str)
    obs["dataset"] = obs["dataset"].astype(str)
    uuids_dict = uuids_df.set_index("uuid").to_dict(orient="index")
    for index, row in obs.iterrows():
        dataset_value = row["dataset"]
        if dataset_value in uuids_dict:
            for key, value in uuids_dict[dataset_value].items():
                obs.at[index, key] = value
    del obs["Unnamed: 0"]
    return obs


def load_adjacency_matrix_and_labels(
    adjacency_file: Path, label_file: Path, adata: anndata.AnnData
):
    adjacency_matrix = mmread(adjacency_file).tocsc()
    labels = pd.read_csv(
        label_file, header=None, names=["cell_id"], delim_whitespace=True
    )

    adata_cell_ids = adata.obs["ID"].astype(int).to_list()
    filtered_labels = labels[labels["cell_id"].isin(adata_cell_ids)]
    filtered_cell_ids = filtered_labels["cell_id"].values

    label_to_index_map = pd.Series(
        labels.index.values, index=labels["cell_id"].astype(int)
    )
    filtered_indices = label_to_index_map[filtered_cell_ids].values

    # Adjust indices to fit the zero-based indexing
    adjusted_indices = filtered_indices - 1
    filtered_matrix = adjacency_matrix[adjusted_indices, :][:, adjusted_indices]
    return filtered_matrix.tocoo()


def create_block_diag_adjacency_matrices(adjacency_matrices):
    block_diag_matrix = block_diag(adjacency_matrices, format="coo")

    return block_diag_matrix.tocsr()


def main(data_dir: Path, uuids_tsv: Path, tissue: str):
    raw_output_file_name = f"{tissue}_raw.h5ad"
    uuids_df = pd.read_csv(uuids_tsv, sep="\t", dtype=str)
    uuids_list = uuids_df["uuid"].to_list()
    hbmids_list = uuids_df["hubmap_id"].to_list()
    hdf5_files_list = []
    cell_count_files_list = []
    adjacency_matrix_files_list = []
    adjacency_matrix_labels_files_list = []
    cell_centers_files_list = []
    directories = [data_dir / Path(uuid) for uuid in uuids_df["uuid"]]

    for directory in directories:
        if len(listdir(directory)) > 1:
            (
                hdf5_files,
                cell_count_files,
                adjacency_matrix_files,
                adjacency_matrix_labels_files,
                cell_centers_files,
            ) = find_files_by_type(directory)
            hdf5_files_list.extend(hdf5_files)
            cell_count_files_list.extend(cell_count_files)
            adjacency_matrix_files_list.extend(adjacency_matrix_files)
            adjacency_matrix_labels_files_list.extend(adjacency_matrix_labels_files)
            cell_centers_files_list.extend(cell_centers_files)

    # Create the AnnData objects and process adjacency matrices
    adatas = []
    filtered_adjacency_matrices = []
    varms_dict = {}

    for (
        hdf5_file,
        cell_centers_file,
        adjacency_file,
        label_file,
        cell_count_file,
    ) in zip(
        hdf5_files_list,
        cell_centers_files_list,
        adjacency_matrix_files_list,
        adjacency_matrix_labels_files_list,
        cell_count_files_list,
    ):
        adata = create_anndata(
            hdf5_file, tissue, uuids_df, cell_centers_file, cell_count_file, data_dir
        )
        adatas.append(adata)
        # Save the values in .varm
        if adata.varm:
            for key in adata.varm.keys():
                if key not in varms_dict:
                    varms_dict[key] = []
                varms_dict[key].append(adata.varm[key])

        # Load and filter the corresponding adjacency matrix
        filtered_matrix = load_adjacency_matrix_and_labels(
            adjacency_file, label_file, adata
        )
        filtered_adjacency_matrices.append(filtered_matrix)
    for key in varms_dict:
        varms_dict[key] = pd.concat(varms_dict[key], axis=1)

    # Concatenate all AnnData objects into one
    combined_adata = anndata.concat(adatas, join="outer")
    combined_adjacency_matrix = create_block_diag_adjacency_matrices(
        filtered_adjacency_matrices
    )
    combined_adata.obsp["adjacency_matrix"] = combined_adjacency_matrix

    # Make sure the var index matches the varm indices and add to concatenated adata 
    for key in varms_dict:
        varms_dict[key] = varms_dict[key].reindex(combined_adata.var.index, fill_value=np.nan)
    combined_adata.varm["RRID"] = varms_dict["RRID"]
    combined_adata.varm["Uniprot_ID"] = varms_dict["Uniprot_ID"]
    combined_adata.varm["Antibodies_Tsv_ID"] = varms_dict["Antibodies_Tsv_ID"]

    # Add patient metadata to obs
    obs_w_patient_info = add_patient_metadata(combined_adata.obs, uuids_df)
    combined_adata.obs = obs_w_patient_info

    # Generate data product metadata and write AnnData
    creation_time = str(datetime.now())
    data_product_uuid = str(uuid.uuid4())
    total_cell_count = combined_adata.obs.shape[0]
    combined_adata.uns["creation_data_time"] = creation_time
    combined_adata.uns["datasets"] = hbmids_list
    combined_adata.uns["uuid"] = data_product_uuid
    combined_adata.write(raw_output_file_name)

    # Save data product metadata
    file_size = os.path.getsize(raw_output_file_name)
    create_json(
        tissue,
        data_product_uuid,
        creation_time,
        uuids_list,
        hbmids_list,
        total_cell_count,
        file_size,
    )


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
