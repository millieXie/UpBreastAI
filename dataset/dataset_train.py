import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


class Litss_Anomaly_DataSet(Dataset):
    def __init__(self, root, label_csv, size=(96, 96, 96), random_mask_prob=0.25,
                 mode='train', train_ratio=0.8, seed=42):

        self.root = root
        self.size = size
        self.random_mask_prob = random_mask_prob
        self.mode = mode
        self.df_labels = pd.read_excel(label_csv, engine='openpyxl')
        self.filename = []

        benign_cases, dcis_cases = [], []
        for idx, row in self.df_labels.iterrows():
            patient_id = str(row['patient_id'])
            patient_path = os.path.join(root, patient_id)

            if not os.path.isdir(patient_path):
                print(f"⚠️ Warning: Folder for patient {patient_id} not found. Skipping.")
                continue

            DCE_path = os.path.join(patient_path, 'DCE.nii.gz')
            if not os.path.exists(DCE_path):
                print(f"⚠️ Warning: Neither DCE exists for patient {patient_id}. Skipping.")
                continue

            label = int(row['label']) if not pd.isna(row['label']) else 0
            if label == 0:
                benign_cases.append((patient_path, label))
            else:
                dcis_cases.append((patient_path, label))


        random.seed(seed)
        random.shuffle(benign_cases)
        split_idx = int(len(benign_cases) * train_ratio)

        if mode == 'train':
            self.filename = benign_cases[:split_idx]
            print(f"train mode: use {len(self.filename)} DEPD")
        elif mode == 'test':
            test_benign = benign_cases[split_idx:]
            self.filename = test_benign + dcis_cases
            print(f"test:  {len(test_benign)} DEPD and {len(dcis_cases)}DCIS")
        elif mode == 'all':
            self.filename = benign_cases + dcis_cases
            print(f"all: use {len(benign_cases)} DEPD and {len(dcis_cases)} DCIS")
        else:
            raise ValueError("mode must 'train'、'test' or 'all'")

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        file, label = self.filename[index]
        shape = self.size

        DCE= self.load_or_default(os.path.join(file, 'DCE.nii.gz'), shape)
        DWI = self.load_or_default(os.path.join(file, 'DWI.nii.gz'), shape)
        ADC = self.load_or_default(os.path.join(file, 'ADC.nii.gz'), shape)

        # ========= data enhancement(only train) =========
        if self.mode == 'train':
            DCE, ADC, DWI = self.augment(DCE, ADC, DWI)

        input_features = np.stack([DCE, ADC, DWI], axis=0)
        input_features = torch.from_numpy(input_features).float()
        label = torch.tensor(label, dtype=torch.long)

        return input_features, label

    # ---------------- 工具函数 ----------------
    def load(self, file):
        return sitk.GetArrayFromImage(sitk.ReadImage(file))

    def load_or_default(self, file, shape):
        img = self.load(file).astype(np.float32)
        return img

    def resize_img(self, img, shape):
        sitk_img = sitk.GetImageFromArray(img)
        original_size = sitk_img.GetSize()  # (W,H,D)
        original_spacing = sitk_img.GetSpacing()

        new_size = [int(s) for s in shape[::-1]]
        new_spacing = [
            original_spacing[i] * (original_size[i] / new_size[i])
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())

        resized_img = resampler.Execute(sitk_img)
        return sitk.GetArrayFromImage(resized_img).astype(np.float32)


    def augment(self, DCE, ADC, DWI):
        if random.random() > 0.5:
            DCE, ADC, DWI= [np.flip(x, axis=1) for x in [DCE, ADC, DWI]]
        if random.random() > 0.5:
            DCE, ADC, DWI = [np.flip(x, axis=2) for x in [DCE, ADC, DWI]]
        if random.random() > 0.5:
            k = random.randint(1, 3)
            DCE, ADC, DWI = [np.rot90(x, k=k, axes=(1, 2)) for x in [DCE, ADC, DWI]]
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, DCE.shape)
            DCE = np.clip(DCE + noise, 0, 1)
            ADC = np.clip(ADC + noise, 0, 1)
            DWI = np.clip(DWI + noise, 0, 1)

        return DCE, ADC, DWI
