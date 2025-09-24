#!/usr/bin/env python3
import json
import os
from argparse import ArgumentParser
from pathlib import Path


def get_uuid(data_product_metadata):
    with open(data_product_metadata, "r") as json_file:
        metadata = json.load(json_file)
    uuid = metadata["Data Product UUID"]
    return uuid


def upload_to_vm(metadata_json, uuid):
    os.system(
        f"scp {metadata_json} /opt/repositories/vm024-dev/data-products-ui/pipeline_outputs/{uuid}.json"
    )


def main(metadata_json):
    uuid = get_uuid(metadata_json)
    upload_to_vm(metadata_json, uuid)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--data_product_metadata", type=Path)
    args = p.parse_args()

    main(args.data_product_metadata)
