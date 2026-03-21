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

#### Option 1 (Recommended: Conda)

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
```

#### Option 2 (pip)

```bash
pip install -r requirements.txt
```

---

### 2. Prepare Directories

```bash
mkdir -p outputs
mkdir -p checkpoints_best
mkdir -p checkpoints_periodic
```

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

```bash
CUDA_VISIBLE_DEVICES=1 python -m train \
experiment=hg38/hg38 \
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

- Uppercase conversion  
- Replace T → U  
- Keep {A, U, C, G}  
- Remove invalid sequences  

### Split

| Split | Ratio |
|------|------|
| Train | 80% |
| Val   | 10% |
| Test  | 10% |

---

## 3. Tokenization

```yaml
dataset.tokenizer_name = char
dataset.kmer = 1
dataset.frame = 0
```

- Character-level tokenizer  
- EOS token appended  

---

## 4. Model

```yaml
model = caduceus
```

### Architecture

- Mamba (State Space Model)  
- Bidirectional  
- Memory-augmented  

### Key Parameters

| Parameter  | Value |
|-----------|------|
| d_model   | 768  |
| n_layer   | 12   |
| vocab_size| 12   |

---

## 5. Training Objective

```yaml
dataset.mlm = true
dataset.mlm_probability = 0.15
```

Masking strategy:

- 80% → mask  
- 10% → random token  
- 10% → unchanged  

---

## 6. Optimization

### Batch

| Setting        | Value |
|---------------|------|
| batch size     | 16   |
| accumulation   | 16   |
| global batch   | 256  |

### Optimizer

- AdamW  
- Learning rate = 8e-5  
- Weight decay = 0.01  

### Scheduler

- Cosine warmup (timm)  
- Warmup steps = 4000  

---

## 7. Training Procedure

- Train on full dataset  
- Validate every epoch  
- Save checkpoints  

---

## 8. Checkpointing ⭐

### Best Checkpoint

- Metric: `val/loss`  

```
./checkpoints_best/val/loss.ckpt
```

### Periodic Checkpoint

```
every_n_epochs = max_epochs // 10
```

Saved to:

```
./checkpoints_periodic/
```

---

## 9. Training Behavior

- 1 epoch ≈ 1528 steps  
- global_step = optimizer updates  
- 1 step = 16 batches  

---

## 10. Important Notes

- Not a classification task  
- Tokenizer must match between training and inference  
- `experiment=hg38/hg38` is used as configuration template  

---

## 11. Acknowledgements

Based on the original **Caduceus** repository.

---

## 12. Citation

```bibtex
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff et al.},
  year={2024}
}
```

---

## 📢 Install Mamba (Required)

```bash
pip install mamba_ssm
```

If installation fails, check:

- Python 3.10  
- PyTorch 2.2.x  
- CUDA 12.x  # mamba-for-RNA
