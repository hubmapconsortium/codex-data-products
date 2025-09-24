#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: Pipeline for concatenating CODEX datasets into H5AD object

requirements:
  ScatterFeatureRequirement: {}

inputs: 
    data_directory:
        label: "Path to directory containing cell by gene and cell by bin files"
        type: Directory
    
    uuids_file:
        label: "Path to a file containing a list of uuids and other metadata for the dataset to be indexed"
        type: File
    
    tissue:
        label: "Two letter tissue type code"
        type: string?

outputs:
    h5mu_file:
        type: File
        outputSource: concatenate/h5mu_file
    
    metadata_json:
        type: File
        outputSource: concatenate/metadata_json

steps:

    - id: concatenate
      in: 
        - id: data_directory
          source: data_directory
        - id: uuids_file
          source: uuids_file
        - id: tissue
          source: tissue
    
      out:
        - h5mu_file
        - metadata_json
      run: steps/concatenate.cwl
      label: "Concatenates out.hdf files in directory"

    - id: upload
      in: 
        - id: h5mu_file
          source: concatenate/h5ad_file
        - id: metadata_json
          source: concatenate/metadata_json

      out:
        - finished_text
      run: steps/upload.cwl
      label: "Uploads the pipeline outputs to s3"
      