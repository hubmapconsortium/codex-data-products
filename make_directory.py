#!/usr/bin/env python3
import json
from antibodies_tsv_util import find_antibodies_meta
from argparse import ArgumentParser
from os import fspath, walk
from pathlib import Path
from subprocess import check_call

import pandas as pd


def find_files(directory, patterns):
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            for pattern in patterns:
                if filepath.match(pattern):
                    return filepath


def find_file_pairs(directory):
    hdf5_pattern = ["out.hdf5"]
    
    cell_count_pattern = [
        "reg1_stitched_expressions.ome.tiff-cell_channel_total.csv", 
        "reg001_expr.ome.tiff-cell_channel_total.csv"
    ]
    adjacency_matrix_pattern = [
        "reg1_stitched_expressions.ome.tiff_AdjacencyMatrix.mtx", 
        "reg001_expr.ome.tiff_AdjacencyMatrix.mtx"
    ]
    adjacency_matrix_labels_pattern = [
        "reg1_stitched_expressions.ome.tiff_AdjacencyMatrixRowColLabels.txt", 
        "reg001_expr.ome.tiff_AdjacencyMatrixRowColLabels.txt"
    ]
    cell_centers_pattern = [
        "reg1_stitched_expressions.ome.tiff-cell_centers.csv", 
        "reg001_expr.ome.tiff-cell_centers.csv"
    ]

    hdf5_file = find_files(directory, hdf5_pattern)
    cell_count_file = find_files(directory, cell_count_pattern)
    adjacency_matrix_file = find_files(directory, adjacency_matrix_pattern)
    adjacency_matrix_labels_file = find_files(directory, adjacency_matrix_labels_pattern)
    cell_centers_file = find_files(directory, cell_centers_pattern)
    
    return (
        hdf5_file, 
        cell_count_file, 
        adjacency_matrix_file, 
        adjacency_matrix_labels_file, 
        cell_centers_file
    )


def get_input_directory(data_directory, uuid):
    public_directory = data_directory / "public" / uuid
    if public_directory.exists():
        return public_directory
    else:
        consortium_directory = data_directory / "consortium"
        if consortium_directory.exists():
            for subdir in consortium_directory.iterdir():
                consortium_subdir = subdir / uuid
                if consortium_subdir.exists():
                    return consortium_subdir
                

def copy_file(file, files_directory):
    check_call(
        f"cp {fspath(file)} {files_directory}/{file.name}",
        shell=True,
    )


def find_parent_file(input_directory, files_directory):
    antibodies_file = find_antibodies_meta(input_directory)
    if antibodies_file:
        print(f"Antibodies TSV: {antibodies_file}")
        copy_file(antibodies_file, files_directory)
    else:
        print(f"No antibodies TSV found in {input_directory}")


def find_processed_files(uuids_df, files_base_directory, data_directory):
    for _, row in uuids_df.iterrows():
        uuid = row['uuid']
        immediate_descendant_ids = row['immediate_descendant_ids']

        files_directory = files_base_directory / uuid
        files_directory.mkdir(parents=True, exist_ok=True)

        input_directory = get_input_directory(data_directory, uuid)

        if pd.notna(immediate_descendant_ids):  # If there are immediate descendants
            find_parent_file(input_directory, files_directory)
        else:
            input_files = find_file_pairs(input_directory)
            if input_files == (None, None, None, None, None):
                print("No input files in: ", input_directory)
                continue
            print("Input directory:", input_directory)
            print("Input files:", input_files)
            for input_file in input_files:
                copy_file(input_file, files_directory)


def main(data_directory: Path, uuids_file: Path, tissue: str):
    uuids_df = pd.read_csv(uuids_file, sep="\t")
    uuids_df = uuids_df.dropna(subset=["uuid"])  # Ensure UUIDs are valid

    files_base_directory = Path(f"{tissue}_files")
    files_base_directory.mkdir(exist_ok=True)

    find_processed_files(uuids_df, files_base_directory, data_directory)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("data_directory", type=Path)
    p.add_argument("uuids_file", type=Path)
    p.add_argument("tissue", type=str)

    args = p.parse_args()

    main(args.data_directory, args.uuids_file, args.tissue)
