#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from os import fspath, walk
from pathlib import Path
from subprocess import check_call

import pandas as pd


def find_file(directory, filename="out.hdf5"):
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for file in filenames:
            if file == filename:
                return dirpath / file
    return None


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
    return None


def main(data_directory: Path, uuids_file: Path, tissue: str):
    uuids = pd.read_csv(uuids_file, sep="\t")["uuid"].dropna()  # Load UUIDs from file
    h5ads_base_directory = Path(f"{tissue}_h5ads")
    h5ads_base_directory.mkdir(exist_ok=True)  # Create base directory if it doesn't exist
    
    for uuid in uuids:
        h5ads_directory = h5ads_base_directory / uuid
        h5ads_directory.mkdir(parents=True, exist_ok=True)  # Create UUID-specific directory
        
        input_directory = get_input_directory(data_directory, uuid)  # Get the directory for the UUID
        
        if input_directory:
            input_file = find_file(input_directory)  # Find the 'out.hdf5' file
            if input_file:
                print(f"Copying file {input_file} to {h5ads_directory}")
                check_call(
                    f"cp {fspath(input_file)} {h5ads_directory}/{input_file.name}",
                    shell=True,
                )
            else:
                print(f"'out.hdf5' not found in {input_directory}")
        else:
            print(f"No input directory found for UUID: {uuid}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("data_directory", type=Path)
    p.add_argument("uuids_file", type=Path)
    p.add_argument("tissue", type=str)

    args = p.parse_args()

    main(args.data_directory, args.uuids_file, args.tissue)
