#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import json
import os


def set_access_keys(access_key_id, secret_access_key):
    os.system(f"aws configure set aws_access_key_id \"{access_key_id}\"")
    os.system(f"aws configure set aws_secret_access_key \"{secret_access_key}\"")


def upload_file_to_s3(local_file, uuid):
    bucket_path = f"s3://hubmap-data-products/{uuid}/"
    file_size = os.path.getsize(local_file)
    os.system(f"aws s3 cp \"{local_file}\" \"{bucket_path}{local_file.name}\" --expected-size \"{file_size}\"")


def upload_files_to_s3(file_list, uuid):
    for file in file_list:
        upload_file_to_s3(file, uuid)


def get_uuid(metadata_json):
    with open(metadata_json) as json_file:
        metadata = json.load(json_file)
    uuid = metadata["Data Product UUID"]
    return uuid


def main(h5ad_file, data_product_metadata, access_key_id, secret_access_key):
    set_access_keys(access_key_id, secret_access_key)
    uuid = get_uuid(data_product_metadata)
    files = [h5ad_file, data_product_metadata]
    upload_files_to_s3(files, uuid)
    f = open("finished.txt", "w")
    f.write("cwl wants an output file for this step")
    f.close()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("h5ad_file", type=Path)
    p.add_argument("metadata_json", type=Path)
    p.add_argument("access_key_id", type=str)
    p.add_argument("secret_access_key", type=str)
    args= p.parse_args()

    main(args.h5ad_file, args.metadata_json, args.access_key_id, args.secret_access_key)