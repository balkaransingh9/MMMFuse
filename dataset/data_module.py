import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .physio_data import PhysioData
from .text_data import TextData
from .ecg_data import ECGData
from .multimodal_data import MultimodalData

from .collate.ecg_collate import ECGCollate
from .collate.multimodal_collate import MultimodalCollate
from .collate.physio_collate import PhysioCollate
from .collate.text_collate import TextCollate

from .utils.normaliser import med_normaliser
from .utils.build_vocab import build_vocab

from transformers import AutoTokenizer
from .utils.med_tokenizer import MedTokenizer

"""
For Unimodal Data
"""
class UnimodalDataModule(pl.LightningDataModule):
    def __init__(self, listfile, task_type='phenotype', modality='physio',
                 lmdb_path = '', normaliser = None,
                 batch_size=64, num_workers=4):
        super().__init__()
        self.listfile = listfile
        self.normaliser = normaliser
        self.lmdb_path = lmdb_path

        if task_type == 'phenotype':
            self.task_type = task_type
        elif task_type == 'in_hospital_mortality':
            self.task_type = task_type
        elif task_type == 'length_of_stay':
            self.task_type = task_type
        else:
            raise ValueError("Task type not supported!")

        if modality == 'physio':
            self.modality = modality
            self.collate_fn = PhysioCollate(task_type=self.task_type)
        elif modality == 'ecg':
            self.modality = modality
            self.collate_fn = ECGCollate(task_type=self.task_type)
        elif modality == 'text':
            self.modality = modality
            self.collate_fn = TextCollate(task_type=self.task_type)
        else:
            raise ValueError("Modality not supported!")

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Called on every GPU separately; stage can be 'fit' or 'test'
        if stage in (None, 'fit'):
            if self.modality == 'physio':
                self.train_ds = PhysioData(self.listfile, task_type=self.task_type, split='train',
                                           normaliser_physio=self.normaliser, lmdb_path_physio=self.lmdb_path)
            if self.modality == 'ecg':
                self.train_ds = ECGData(self.listfile, task_type=self.task_type, split='train',
                                        normaliser_ecg=self.normaliser, lmdb_path_ecg=self.lmdb_path)
            if self.modality == 'text':
                self.train_ds = TextData(self.listfile, task_type=self.task_type, split='train',
                                         lmdb_path_text=self.lmdb_path)

        if stage in (None, 'test'):
            if self.modality == 'physio':
                self.test_ds = PhysioData(self.listfile, task_type=self.task_type, split='test',
                                           normaliser_physio=self.normaliser, lmdb_path_physio=self.lmdb_path)
            if self.modality == 'ecg':
                self.test_ds = ECGData(self.listfile, task_type=self.task_type, split='test',
                                        normaliser_ecg=self.normaliser, lmdb_path_ecg=self.lmdb_path)
            if self.modality == 'text':
                self.test_ds = TextData(self.listfile, task_type=self.task_type, split='test',
                                         lmdb_path_text=self.lmdb_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

"""
For Multimodal Data
"""
class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, listfile, task_type='phenotype', modalities = ['physio','ecg','text'],
                 lmdb_path_physio = '', lmdb_path_ecg = '',
                 lmdb_path_text = '', lmdb_path_medicine = '',
                 text_model_name = 'nlpie/tiny-clinicalbert', text_max_len = 512,
                 normaliser_physio = None, normaliser_ecg = None, batch_size=64, num_workers=4):
        super().__init__()
        self.listfile = listfile
        self.modalities = modalities
       
        self.lmdb_path_physio = lmdb_path_physio
        self.lmdb_path_ecg = lmdb_path_ecg
        self.lmdb_path_text = lmdb_path_text
        self.lmdb_path_medicine = lmdb_path_medicine

        self.normaliser_physio = normaliser_physio
        self.normaliser_ecg = normaliser_ecg

        if task_type == 'phenotype':
            self.task_type = task_type
        elif task_type == 'in_hospital_mortality':
            self.task_type = task_type
        elif task_type == 'length_of_stay':
            self.task_type = task_type
        else:
            raise ValueError("Task type not supported!")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len

        self.mednorm = med_normaliser(self.listfile, self.lmdb_path_medicine)
        self.label_vocab = build_vocab(self.lmdb_path_medicine, 'label')
        self.unit_vocab = build_vocab(self.lmdb_path_medicine, 'amount_std_uom')
        self.cat_vocab = build_vocab(self.lmdb_path_medicine, 'ordercategoryname')

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.med_tokenizer = MedTokenizer(self.label_vocab, self.unit_vocab,
                                          self.cat_vocab, self.mednorm)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Called on every GPU separately; stage can be 'fit' or 'test'
        if stage in (None, 'fit'):
            self.train_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='train',
                                           normaliser_physio=self.normaliser_physio, normaliser_ecg=self.normaliser_ecg, 
                                           lmdb_path_physio=self.lmdb_path_physio, lmdb_path_ecg=self.lmdb_path_ecg,
                                           lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine)

        if stage in (None, 'test'):
            self.test_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='test',
                                           normaliser_physio=self.normaliser_physio, normaliser_ecg=self.normaliser_ecg, 
                                           lmdb_path_physio=self.lmdb_path_physio, lmdb_path_ecg=self.lmdb_path_ecg,
                                           lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=MultimodalCollate(modalities=self.modalities,
                                         text_tokenizer=self.text_tokenizer,
                                         med_tokenizer=self.med_tokenizer,
                                         task_type=self.task_type, text_max_len=self.text_max_len)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=MultimodalCollate(modalities=self.modalities,
                                         text_tokenizer=self.text_tokenizer,
                                         med_tokenizer=self.med_tokenizer,
                                         task_type=self.task_type, text_max_len=self.text_max_len)
        )
