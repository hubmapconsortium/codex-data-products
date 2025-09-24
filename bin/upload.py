#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import json
import os


def copy_file_to_vm(local_file, uuid):
    os.system(f'mkdir /hive/hubmap/data/public/cmu-organ-data-products/{uuid}')
    os.system(
        f'scp {local_file} /hive/hubmap/data/public/cmu-organ-data-products/{uuid}/{local_file.name}'
    )


def copy_files_to_vm(file_list, uuid):
    for file in file_list:
        copy_file_to_vm(file, uuid)


def get_uuid(metadata_json):
    with open(metadata_json) as json_file:
        metadata = json.load(json_file)
    uuid = metadata["Data Product UUID"]
    return uuid


def main(
    raw_h5mu,
    data_product_metadata,
):
    uuid = get_uuid(data_product_metadata)
    files = [raw_h5mu, data_product_metadata]
    copy_files_to_vm(files, uuid)
    f = open("finished.txt", "w")
    f.write("cwl wants an output file for this step")
    f.close()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("annotated_raw_h5ad_file", type=Path)
    p.add_argument("processed_h5ad_file", type=Path)
    p.add_argument("umap_png", type=Path)
    p.add_argument("data_product_metadata", type=Path)
    p.add_argument("zarr_file", type=str)
    args = p.parse_args()

    main(
        args.raw_h5mu,
        args.data_product_metadata,
    )
