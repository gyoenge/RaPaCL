# Dataset Preparation

This document describes how to prepare the dataset required for **RaPaCL-ST**.

---

## 1. Set Hugging Face Token

Some datasets are hosted on Hugging Face Hub.  
You need to set your access token as an environment variable.

```bash
touch .env
```

Add your token:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

(Optional) Prevent committing the token:
```bash
echo ".env" >> .gitignore
```

## 2. Download HEST / HEST-Bench

We use the **HEST** and **HEST-Bench** datasets for spatial transcriptomics prediction.

Run the following command to download the data:

```bash
python -m src.data.download_hest \
  --config configs/data/download_hest.yaml
```

The configuration file specifies:

- dataset version (HEST / HEST-Bench)
- target directory
- download options

## 3. Extract Gene List

To define the prediction targets, we extract a gene list (e.g., highly variable genes).

```bash 
python -m src.data.extract_genes \
  --config configs/data/extract_genes.yaml
```

This step typically:

- loads gene expression matrices
- selects genes based on criteria (e.g., HVG)
- saves a JSON file for downstream training

## Output Structure

After preparation, the dataset is organized as follows:

    data/
    ├── hest/
    ├── hest_bench/
    ├── genes/
    │   └── {criteria}_{num_genes}genes.json

## Notes

- Ensure sufficient disk space before downloading datasets.
- The Hugging Face token is required for private or gated datasets.
- All preprocessing steps are controlled via YAML configuration files for reproducibility.

