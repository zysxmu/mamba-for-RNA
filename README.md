# Caduceus-RNA

**Memory-Augmented Bidirectional Mamba for Mixed RNA Representation Learning**

---

## 1. Overview

This project implements **Caduceus-RNA**, a memory-augmented bidirectional sequence model for RNA representation learning.

The model is trained using **Masked Language Modeling (MLM)** on a **mixed RNA dataset**:

- Coding RNA (TXT)
- Non-coding RNA (FASTA)

### Key Features

- Bidirectional Mamba architecture
- Memory-augmented sequence modeling
- Mixed RNA pretraining
- Character-level biological tokenizer

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

---

## 2.⚠️ Special Dependencies

This project depends on two **non-standard packages**:

### 1. causal-conv1d

- Required by Mamba architecture
- May fail if built from source
- Recommended for manual installatio

✅ Recommended install:

```bash
pip install causal-conv1d==1.2.0.post2 --no-build-isolation
or
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

---

### 2. mamba_ssm

- Core implementation of Mamba (State Space Model)
- Required for model to run
- Recommended for manual installation

```bash
pip install mamba_ssm/mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

If installation fails, check:

- Python = 3.10
- PyTorch = 2.2.x
- CUDA = 12.x

---

### 3. Prepare Dataset

Place files under `data/`:

```
data/
├── data-random_15K_sequences.txt
├── rnacentral_small_ATCG_only.fasta
```

---

### 4. Run Training

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python -m train ...
```

---

### ⭐ Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m train \
trainer.devices=2 \
trainer.accelerator=gpu \
...
```

📌 Notes:

- `CUDA_VISIBLE_DEVICES=0,1` → select GPUs
- `trainer.devices=2` → number of GPUs
- Uses **DDP (DistributedDataParallel)** automatically
- Make sure batch size fits GPU memor

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m train \
experiment=hg38/hg38 \
trainer.devices=2 \
trainer.accelerator=gpu \
dataset.dataset_name=mixed_rna \
+dataset.text_file=data/data-random_15K_sequences.txt \
+dataset.rna_fasta_file=data/rnacentral_small_ATCG_only.fasta \
dataset.tokenizer_name=char \
+dataset.kmer=1 \
+dataset.frame=0 \
dataset.max_length=1024 \
dataset.batch_size=16 \
dataset.batch_size_eval=64 \
dataset.mlm=true \
dataset.mlm_probability=0.15 \
model=caduceus \
model.config.d_model=768 \
model.config.n_layer=12 \
model.config.vocab_size=12 \
model.config.bidirectional=true \
model.config.bidirectional_strategy=add \
model.config.bidirectional_weight_tie=true \
model.config.rcps=false \
+model.config.pad_token_id=4 \
optimizer.lr=8e-5 \
optimizer.weight_decay=0.01 \
'optimizer.betas=[0.9,0.98]' \
trainer.precision=16 \
trainer.gradient_clip_val=0.5 \
trainer.max_epochs=50 \
trainer.max_steps=20000 \
trainer.limit_val_batches=1.0 \
+trainer.val_check_interval=100 \
trainer.num_sanity_val_steps=0 \
scheduler._name_=cosine_warmup_timm \
scheduler.t_initial=30000 \
scheduler.warmup_t=4000 \
scheduler.lr_min=2e-5 \
scheduler.warmup_lr_init=1e-6 \
wandb=null \
train.monitor=val/loss \
train.mode=min \
callbacks.model_checkpoint.monitor=val/loss \
callbacks.model_checkpoint.mode=min \
callbacks.model_checkpoint.dirpath=./checkpoints_best \
callbacks.model_checkpoint.filename=val/loss \
callbacks.periodic_checkpoint.dirpath=./checkpoints_periodic
```

---

### 5. Outputs

```
checkpoints_best/val/loss.ckpt
checkpoints_periodic/epochXX.ckpt
outputs/
```

---

### 6. Resume Training (Optional)

```bash
train.ckpt=checkpoints/last.ckpt
```

---

### 7. Final Evaluation

- Automatically loads best checkpoint
- Runs test set evaluation

---

## 2. Dataset

### Data Sources

- Coding RNA (TXT)
- Non-coding RNA (FASTA)

### Preprocessing

- Uppercase
- T → U
- Keep {A, U, C, G}
- Remove invalid sequences

### Split

| Split | Ratio |
| --- | --- |
| Train | 80% |
| Val | 10% |
| Test | 10% |

---

## 3. Tokenization

```yaml
dataset.tokenizer_name = char
dataset.kmer = 1
dataset.frame = 0
```

- Character-level tokenizer
- EOS appended

---

## 4. Model

```yaml
model = caduceus
```

### Architecture

- Mamba (state space model)
- Bidirectional
- Memory-augmented

### Key Parameters

| Parameter | Value |
| --- | --- |
| d_model | 768 |
| n_layer | 12 |
| vocab_size | 12 |

---

## 5. Training Objective

```yaml
dataset.mlm = true
dataset.mlm_probability = 0.15
```

Masking:

- 80% → mask
- 10% → random
- 10% → unchanged

---

## 6. Optimization

### Batch

| Setting | Value |
| --- | --- |
| batch size | 16 |
| accumulation | 16 |
| global batch | 256 |

---

### Optimizer

- AdamW
- LR = 8e-5
- weight_decay = 0.01

---

### Scheduler

- Cosine warmup (timm)
- warmup = 4000

---

## 7. Training Procedure

- Train → full dataset
- Validate → each epoch
- Save checkpoints

---

## 8. Checkpointing ⭐

### Best Checkpoint

- Metric: `val/loss`
- Path:

```
./checkpoints_best/val/loss.ckpt
```

---

### Periodic Checkpoint (Custom)

We implement **epoch-based periodic saving**:

```python
every_n_epochs = max_epochs // 10
```

Example:

| max_epochs | interval |
| --- | --- |
| 50 | 5 |
| 100 | 10 |
| 200 | 20 |

Saved to:

```
./checkpoints_periodic/
```

Format:

```
epoch05.ckpt
epoch10.ckpt
...
```

Notes:

- `save_top_k = -1` → save all periodic checkpoints
- No metric filtering

---

## 9. Training Behavior

- 1 epoch ≈ 1528 steps
- global_step = optimizer updates
- 1 step = 16 batches

Example:

```
Epoch 0 → global_step ≈ 93
```

---

## 10. Important Notes

⚠ Not a classification task

⚠ Tokenizer must match between training and inference

⚠ `experiment=hg38/hg38` is used as config template

---

## 11. Acknowledgements

Based on the original **Caduceus** repository.

---

## 12. Citation

```
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff et al.},
  year={2024}
}
```

---