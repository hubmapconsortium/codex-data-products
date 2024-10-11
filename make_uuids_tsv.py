#!/usr/bin/env python3
import json
from argparse import ArgumentParser

import pandas as pd
import requests
import yaml

organ_types_yaml_file = "bin/organ_types.yaml"


def get_uuids(organ_name_mapping: dict, organ: str = None):
    organ_code_mapping = {organ_name_mapping[key]: key for key in organ_name_mapping}
    
    # Define the query payload based on organ
    if organ is None:
        query_payload = {
            "from": 0,
            "size": 10000,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"dataset_type": "CODEX"}},
                        {"match": {"data_access_level": "public"}}
                    ]
                }
            }
        }
    else:
        query_payload = {
            "from": 0,
            "size": 10000,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"dataset_type": "CODEX"}},
                        {"match": {"data_access_level": "public"}},
                        {"match": {"origin_samples.organ": organ_code_mapping[organ]}}
                    ]
                }
            }
        }

    # Make the request
    url = "https://search.api.hubmapconsortium.org/v3/search"
    response = requests.post(url, json=query_payload)
    print("Response status code: ", response.status_code)

    # Handle a successful response
    if response.status_code == 200:
        return process_response(response)
    
    # Handle 303 redirection
    elif response.status_code == 303:
        redirection_url = response.text.strip()  # Get redirection URL from response body
        print("Following redirection URL: ", redirection_url)
        
        # Make a request to the redirection URL
        redirection_response = requests.get(redirection_url)
        if redirection_response.status_code == 200:
            return process_response(redirection_response)
    
    # Handle other error responses
    else:
        print(f"Error {response.status_code}: {response.text}")
        return [], [], []


def process_response(response):
    """
    Helper function to process the JSON response and extract UUIDs, HubMAP IDs, and donor metadata.
    """
    data = response.json()
    hits = data.get("hits", {}).get("hits", [])
    
    uuids = []
    hubmap_ids = []
    donor_metadata_list = []

    # Loop through each dataset (hit) and extract the relevant information
    for hit in hits:
        source = hit["_source"]
        uuids.append(source["uuid"])
        hubmap_ids.append(source["hubmap_id"])
        
        # Attempt to extract donor metadata, if available
        donor_metadata = source.get("donor", {}).get("metadata", {})
        donor_metadata_list.append(extract_donor_metadata(donor_metadata))

    return uuids, hubmap_ids, donor_metadata_list


def extract_donor_metadata(metadata):
    """
    Extract donor metadata, including age, sex, height, weight, and other relevant information.
    """
    donor_info = {
        "age": None,
        "sex": None,
        "height": None,
        "weight": None,
        "bmi": None,
        "cause_of_death": None,
        "race": None
    }

    for item in metadata.get('organ_donor_data', []):
        concept = item.get('grouping_concept_preferred_term')
        value = item.get('data_value')
        
        if concept == "Age":
            donor_info["age"] = value
        elif concept == "Sex":
            donor_info["sex"] = item.get('preferred_term')
        elif concept == "Height":
            donor_info["height"] = value
        elif concept == "Weight":
            donor_info["weight"] = value
        elif concept == "Body mass index":
            donor_info["bmi"] = value
        elif concept == "Cause of death":
            donor_info["cause_of_death"] = item.get('preferred_term')
        elif concept == "Race":
            donor_info["race"] = item.get('preferred_term')

    for item in metadata.get('living_donor_data', []):
        concept = item.get('grouping_concept_preferred_term')
        value = item.get('data_value')
        if concept == "Age":
            donor_info["age"] = value
        elif concept == "Sex":
            donor_info["sex"] = item.get('preferred_term')
        elif concept == "Height":
            donor_info["height"] = value
        elif concept == "Weight":
            donor_info["weight"] = value
        elif concept == "Body mass index":
            donor_info["bmi"] = value
        elif concept == "Cause of death":
            donor_info["cause_of_death"] = item.get('preferred_term')
        elif concept == "Race":
            donor_info["race"] = item.get('preferred_term')

    return donor_info




def main(tissue_type: str):
    organ_dict = yaml.load(open(organ_types_yaml_file), Loader=yaml.BaseLoader)
    for key in organ_dict:
        organ_dict[key] = organ_dict[key]["description"]
    if tissue_type not in organ_dict.values():
        print(f"Tissue {tissue_type} not found ")
        tissue_type = None
    uuids_list, hubmap_ids_list, donor_metadata = get_uuids(organ_dict, tissue_type)
    uuids_df = pd.DataFrame()
    uuids_df["uuid"] = pd.Series(uuids_list, dtype=str)
    uuids_df["hubmap_id"] = pd.Series(hubmap_ids_list, dtype=str)
    donor_metadata_df = pd.DataFrame(donor_metadata)
    result_df = pd.concat([uuids_df, donor_metadata_df], axis=1)
    key_for_tissue = [key for key, value in organ_dict.items() if value == tissue_type]
    if key_for_tissue:
            output_file_name = f"{key_for_tissue[0].lower()}.tsv"
    else:
        output_file_name = "codex.tsv"
    print(result_df)
    result_df.to_csv(output_file_name, sep="\t")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("tissue_type", type=str, nargs="?", help="Type of tissue (optional)")

    args = p.parse_args()

    main(args.tissue_type)
