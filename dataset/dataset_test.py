import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


class Litss_Anomaly_DataSet(Dataset):
    def __init__(self, root, label_csv, size=(96, 96, 96), random_mask_prob=0.25,
                 mode='train', train_ratio=1.0, seed=42):

        self.root = root
        self.size = size
        self.random_mask_prob = random_mask_prob
        self.mode = mode
        self.df_labels = pd.read_excel(label_csv, engine='openpyxl')

        self.samples = []

        benign_cases, dcis_cases = [], []

        for idx, row in self.df_labels.iterrows():
            patient_id = str(row['patient_id'])
            patient_path = os.path.join(root, patient_id)

            if not os.path.isdir(patient_path):
                print(f"⚠️ Warning: Folder for patient {patient_id} not found. Skipping.")
                continue

            label = int(row['label']) if not pd.isna(row['label']) else 0

            sample_tuple = (patient_path, label, patient_id)

            if label == 0:
                benign_cases.append(sample_tuple)
            else:
                dcis_cases.append(sample_tuple)

        random.seed(seed)
        random.shuffle(benign_cases)
        split_idx = int(len(benign_cases) * train_ratio)

        if mode == 'test':
            test_benign = benign_cases[split_idx:]
            self.samples = test_benign + dcis_cases
            print(f"test: use {len(test_benign)} DPED and {len(dcis_cases)} DCIS")

        else:
            self.samples = benign_cases + dcis_cases
            print(f"all: use {len(benign_cases)} DEPD and {len(dcis_cases)} DCIS")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        #it should not be enhancement in test data
        patient_path, label, patient_id = self.samples[index]
        shape = self.size

        DCE = self.load_or_default(os.path.join(patient_path, 'DCE.nii.gz'), shape)
        ADC = self.load_or_default(os.path.join(patient_path, 'ADC.nii.gz'), shape)
        DWI = self.load_or_default(os.path.join(patient_path, 'DWI.nii.gz'), shape)

        input_features = np.stack([DCE, ADC, DWI], axis=0)
        input_features = torch.from_numpy(input_features).float()

        label = torch.tensor(label, dtype=torch.long)

        return input_features, label, patient_id

    def load(self, file):
        return sitk.GetArrayFromImage(sitk.ReadImage(file))

    def load_or_default(self, file, shape):
        if os.path.exists(file):
            img = self.load(file).astype(np.float32)
            img = self.resize_img(img, shape)
            return img, 1
        else:
            return np.zeros(shape, dtype=np.float32), 0

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

