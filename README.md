# RaPaCL
RadiomicsFeature-Pathomics Contrastive Learning. 

**RaPaCL (RadiomicsFeature-Pathomics Contrastive Learning)** is a multimodal representation learning framework designed to bridge handcrafted radiomics features and deep pathomics features derived from histopathology images. In this approach, radiomics features extracted from image patches serve as structured, interpretable signals, while deep learning based patch image encoders encode high-dimensional visual representations. RaPaCL leverages contrastive learning to align these two modalities in a shared latent space, encouraging consistency between radiomics-informed characteristics (e.g., texture, heterogeneity) and deep image embeddings. By doing so, the framework aims to enhance the biological relevance and interpretability of learned representations, ultimately improving downstream tasks such as spatial gene expression prediction and tumor characterization in whole-slide images. 

---

## Prepare data

(i) Hugging Face 토큰 환경변수로 설정
```bash
touch .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
echo ".env" >> .gitignore
```

(ii) HEST / HEST-Bench 다운로드
```bash
python -m src.data.download_hest --config configs/data/download_hest.yaml
```

(iii) gene list 추출
```bash
python -m src.data.extract_genes --config configs/data/extract_genes.yaml
```


---

## RaPaCL

### Run RaPaCL

... 

### Pretrain Radiomics TransTab 

```bash
cd RaPaCL/
```

(i) prepare tabular custom data 
- from H5 (HEST-style)
  ```bash
  python -m src.pretrain_transtab.prepare_tabular \
    --config configs/data/prepare_tabular.yaml 
  ```

(ii) pretrain 

- mode explanation: 
  - `train`: contrastive pretraining
  - `eval`: classifier finetuning + test classification metric (accuracy / macro F1 / AUROC)
  - `eval_detailed`: representation quality 평가 전용 (표현 공간 자체 평가)
    - embedding 추출
    - clustering metrics: Silhouette / NMI / ARI
    - UMAP / t-SNE visualization 
    - label cluster visualization
    - radiomics feature representative points
    - feature quantile section visualization

- pretrain with single GPU 
  ```bash
  python -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --distributed false \
    --mode train
  ```

- pretrain with multi GPU 
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --mode train
  ```

- simple evaluation of pretraining
  ```bash 
  python -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --distributed false \
    --mode eval
  ```

- detailed evaluation of pretraining
  ```bash
  python -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --distributed false \
    --mode eval_detailed
  ```

---

## Baselines

### Run stnet

```bash
cd RaPaCL/
```

(i) train
```bash
python -m src.baselines.stnet.run \
  --config configs/stnet.yaml \
  --mode train
```

(ii) eval 
```bash
python -m src.baselines.stnet.run \
  --config configs/stnet.yaml \
  --mode eval
```

(iii) tuning 
```bash
python -m src.baselines.stnet.run \
  --config configs/stnet.yaml \
  --mode tuning
```

### Run Img2Rad

```bash
cd RaPaCL/
```

(i) run both train & eval
```bash
python -m src.baselines.img2rad.main \
  --config configs/img2rad.yaml \
  --mode all
```

(ii) run train or eval individually
```bash
python -m src.baselines.img2rad.main \
  --config configs/img2rad.yaml \
  --mode train
```
```bash
python -m src.baselines.img2rad.main \
  --config configs/img2rad.yaml \
  --mode eval
```

(iii) inspect parquet 
```bash
python -m src.baselines.img2rad.inspect \
  --config configs/img2rad.yaml \
  --mode parquet \
  --show_columns
```

(iv) run ablation studies
- see  `scripts/run_img2rad_*.sh`. 

---

