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

