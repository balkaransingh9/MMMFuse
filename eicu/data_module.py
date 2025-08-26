import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import json 
from pathlib import Path

from .dataset import MultimodalData
from .eicu_collate import MultimodalCollate

from .normaliser import normaliser

"""
For eICU Data
"""
class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, listfile, task_type='mortality', modalities = ['vital','lab','medicine','text'],
                 lmdb_path_vital = '', lmdb_path_lab = '', lmdb_path_medicine = '',
                 csv_path_demographic = '', csv_path_diagnosis = '', csv_path_treatment = '',
                 batch_size=64, num_workers=4):
        
        super().__init__()
        self.listfile = listfile
        self.modalities = modalities
       
        self.lmdb_path_vital = lmdb_path_vital
        self.lmdb_path_lab = lmdb_path_lab
        self.lmdb_path_medicine = lmdb_path_medicine

        if task_type == 'mortality':
            self.task_type = task_type
        elif task_type == 'readmission':
            self.task_type = task_type
        else:
            raise ValueError("Task type not supported!")
        
        self.batch_size = batch_size
        self.num_workers = num_workers


        #static modalities
        self.demographic = pd.read_csv(csv_path_demographic)
        self.diagnosis = pd.read_csv(csv_path_diagnosis)
        self.treatment = pd.read_csv(csv_path_treatment)

        if not self.demographic.empty:
            pid_col = self.demographic["patientunitstayid"]
            self.demographic = self.demographic.drop(columns=["patientunitstayid"]).astype("float32")
            self.demographic.insert(0, "patientunitstayid", pid_col)

        if not self.diagnosis.empty:
            pid_col = self.diagnosis["patientunitstayid"]
            self.diagnosis = self.diagnosis.drop(columns=["patientunitstayid"]).astype("int32")
            self.diagnosis.insert(0, "patientunitstayid", pid_col)

        if not self.treatment.empty:
            pid_col = self.treatment["patientunitstayid"]
            self.treatment = self.treatment.drop(columns=["patientunitstayid"]).astype("int32")
            self.treatment.insert(0, "patientunitstayid", pid_col)

        self.vital_norm = normaliser(self.listfile, lmdb_path=self.lmdb_path_vital)
        self.lab_norm = normaliser(self.listfile, lmdb_path=self.lmdb_path_lab)

    def prepare_data(self):
        pass


    def setup(self, stage=None):
        # Called on every GPU separately; stage can be 'fit' or 'test'
        if stage in (None, 'fit'):
            self.train_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='train',
                                           lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab, lmdb_path_medicine=self.lmdb_path_medicine,
                                           demographic_file = self.demographic, diagnosis_file=self.diagnosis, treatment_file=self.treatment,
                                           vital_norm_stats=self.vital_norm, lab_norm_stats=self.lab_norm)
           
            self.val_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='val',
                                           lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab, lmdb_path_medicine=self.lmdb_path_medicine,
                                           demographic_file = self.demographic, diagnosis_file=self.diagnosis, treatment_file=self.treatment,
                                           vital_norm_stats=self.vital_norm, lab_norm_stats=self.lab_norm)

        if stage in (None, 'test'):
            self.test_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='test',
                                           lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab, lmdb_path_medicine=self.lmdb_path_medicine,
                                           demographic_file = self.demographic, diagnosis_file=self.diagnosis, treatment_file=self.treatment,
                                           vital_norm_stats=self.vital_norm, lab_norm_stats=self.lab_norm)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=MultimodalCollate(modalities=self.modalities, task_type=self.task_type)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=MultimodalCollate(modalities=self.modalities, task_type=self.task_type)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=MultimodalCollate(modalities=self.modalities, task_type=self.task_type)
        )
