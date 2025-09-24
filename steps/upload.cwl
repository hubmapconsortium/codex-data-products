cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-data-products
baseCommand: /opt/upload.py

inputs:
    h5ad_file:
        type: File
        doc: The raw h5ad file
        inputBinding:
            position: 0

    metadata_json:
        type: File
        doc: data product metadata json
        inputBinding:
            position: 1
    
outputs: 
    finished_text:
        type: File
        outputBinding:
            glob: "*.txt"