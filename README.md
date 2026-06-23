# UpBreastAI

UpBreastAI is a PyTorch implementation for mpMRI-based breast anomaly detection. The model learns normal benign patterns from multimodal MRI and uses reconstruction error plus SVDD latent distance to score potential abnormalities.

The current code uses three MRI sequences:

- DCE
- ADC
- DWI

The dataset loader stacks them as:

```python
[DCE, ADC, DWI]
```

## Project Structure

```text
UpBreastAI/
+-- train_new.py                  # training script
+-- test_new.py                   # independent testing script
+-- dataset/
|   +-- dataset_train.py          # training/validation dataset loader
|   +-- dataset_test.py           # testing dataset loader
+-- Model/
|   +-- network_v9fusion_s.py     # MemAE-SVDD hybrid network
|   +-- RUnet.py                  # pretrained autoencoder backbone
|   +-- RUnet_encoder_decoder.py  # pretrained encoder/decoder loading
+-- options/
|   +-- BasicOptions.py           # command-line/default options
|   +-- Options.py
```

## Environment

Main dependencies:

```text
python
torch
numpy
pandas
scikit-learn
scipy
matplotlib
SimpleITK
openpyxl
tqdm
```

Install example:

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib SimpleITK openpyxl tqdm
```

## Data Format

Each patient should have one folder under `--datapath`.

Expected folder layout:

```text
DATA_ROOT/
+-- patient_001/
|   +-- DCE.nii.gz
|   +-- ADC.nii.gz
|   +-- DWI.nii.gz
+-- patient_002/
|   +-- DCE.nii.gz
|   +-- ADC.nii.gz
|   +-- DWI.nii.gz
+-- ...
```
## Training

Edit `train_new.py` first:

```python
label_xlsx = "/input your train data.xlsx"
```

Edit default options in `options/BasicOptions.py` or pass them through command line:

```python
--datapath       # root directory containing patient folders
--checkpoints_dir
--task_name
--gpu_ids
--batch_size
--epoch
--lr
```

Run:

```bash
python train_new.py --datapath /path/to/data --gpu_ids 0 --batch_size 4
```

### Training Strategy

- Stage 1: Train modality-specific reconstruction networks and save the pretrained weights.
- Stage 2: Train the multimodal memory-SVDD anomaly detection model using the pretrained weights.

### Train/Validation Split

`train_new.py` uses a single training Excel file and splits it into training and validation subsets.

Current behavior:

```text
1. Read all samples from label_xlsx.
2. Stratified split by label using train_ratio = (you want).
3. Training set uses only benign samples from the XX% split.
4. Validation set uses all samples from the remaining 1-XX% split.
```

This avoids using the external test set for model selection.

### Training 

Default training settings in `train_new.py`:

```python
memory_size = 300
temperature = 0.05
lambda_recon = 10.0
lambda_diversity = 1e-3
lambda_svdd = 0.1
max_patience = 5
```

The model saves:

```text
checkpoints/<task_name>/latest_model.pth
checkpoints/<task_name>/best_model.pth
```

`best_model.pth` is selected by validation AUC.

## Testing

Edit `test_new.py` before running:

```python
label_csv="/input your test datapath"
fold_dir = os.path.join(r"/input your modelpth")
best_model_path = os.path.join(fold_dir, "your_model(like best_model/latest_model).pth")
```

Run:

```bash
python test_new.py --datapath /path/to/data --gpu_ids 0 --batch_size 4
```

The test script loads the checkpoint, computes anomaly scores, and reports:

```text
AUC
Accuracy
Sensitivity
Specificity
PPV
NPV
F1
MCC
95% confidence intervals
```

Results are saved to:

```text
result/
+-- metrics_with_ci.csv
+-- no_miss_surgery_reduction.csv
+-- score_distribution.png
+-- test_roc.csv
+-- test_roc.png
+-- test_scores.csv
+-- test_final_pred.csv
+-- test_final_pred.xlsx
+-- test_results.xlsx
```

## Notes

- Training uses only benign samples to learn normal patterns.
- Validation should include both benign and abnormal samples.
- Independent test data should only be used in `test_new.py`.
- The anomaly score combines reconstruction error and SVDD distance with `alpha = 0.5`.
- If changing `memory_size` or `temperature`, make sure `train_new.py` and `test_new.py` use matching model settings.
- If loading fails, first check pretrained weight paths in `Model/RUnet_encoder_decoder.py` and checkpoint paths in `test_new.py`.

The original code is intended only for the research experiments of this project. 
