
# RIDG-Diff

Code repository for **RIDG-Diff: Reference-to-Input Discrepancy-Guided Mean-Reverting Diffusion for Multitemporal Cloud Removal**.

The complete code and pretrained models will be released after publication.

---

## Environment

The code was tested with the following environment:

| Component | Version |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.2.1+cu121 |
| torchvision | 0.17.1 |
| CUDA | 12.1 |
| cuDNN | 8.9.2 |

Create the environment:

```bash
conda create -n ridg python=3.10 -y
conda activate ridg
```

Install PyTorch:

```bash
pip install torch==2.2.1 torchvision==0.17.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

### Sen2_MTC_New

Dataset link:

https://github.com/come880412/CTGAN

### MultipleImage

Dataset link:

https://github.com/XavierJiezou/PMAA

---

## Pretrained Models

Pretrained models will be released after publication.

---

## Testing

```bash
python main.py \
    --base configs/example_training/sen2_mtc_new_test_only.yaml \
    --enable_tf32 \
    -t false
```

---

## Citation

Citation information will be updated after publication.

---

