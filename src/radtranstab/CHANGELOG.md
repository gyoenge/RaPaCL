# Previous Readme Checkpoint 

## Pretrain Radiomics TransTab 

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
  python -m src.radtranstab.pipelines.pretrain \
    --config src/radtranstab/configs/idc_hestbench_withshape.yaml \
    --distributed false \
    --mode train
  ```

- pretrain with multi GPU 
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m src.radtranstab.pipelines.pretrain \
    --config src/radtranstab/configs/idc_hestbench_withshape.yaml \
    --mode train

  ```

- simple evaluation of pretraining
  ```bash 
  python -m src.radtranstab.pipelines.pretrain \
    --config src/radtranstab/configs/idc_hestbench_withshape.yaml \
    --distributed false \
    --mode eval
  ```

- detailed evaluation of pretraining
  ```bash
  python -m src.radtranstab.pipelines.pretrain \
    --config src/radtranstab/configs/idc_hestbench_withshape.yaml \
    --distributed false \
    --mode eval_detailed
  ```

---

### Updates

This customized version of TransTab includes:
- Support for **distributed training via PyTorch DDP**
- Support for **multi-class classification beyond binary settings**
- Support for **various tabular column partitioning modes**

