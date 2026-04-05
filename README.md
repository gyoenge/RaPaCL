# RaPaCL
Radiomics-Pathomics Contrastive Learning. 

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

...

---

