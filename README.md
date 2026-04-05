# RaPaCL
Radiomics-Pathomics Contrastive Learning. 

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

(i) train
```bash
python -m src.baselines.img2rad.main \
  --config configs/img2rad.yaml \
  --mode train
```

(ii) eval 
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

---

