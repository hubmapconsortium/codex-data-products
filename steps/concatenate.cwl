cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-data-products
baseCommand: /opt/concatenate.py

inputs:
    data_directory:
        label: "Where the h5ad files are"
        type: Directory
        inputBinding:
            position: 0
    
    uuids_file:
        label: "TSV with metadata"
        type: File
        inputBinding:
            position: 1
    
    tissue:
        label: "Two letter tissue code"
        type: string?
        inputBinding:
            position: 2

outputs:
    h5ad_file:
        type: File
        outputBinding:
            glob: "*.h5ad"
        doc: h5ad file with concatenated codex datasets
    
    metadata_json:
        type: File
        outputBinding: 
            glob: "*.json"
        doc: json containing data product info