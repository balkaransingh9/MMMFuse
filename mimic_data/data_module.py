import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json 

#from .multimodal_data import MultimodalData
from .multimodal_data_missing import MultimodalData

from .collate.multimodal_collate import MultimodalCollate

from .utils.normaliser import med_normaliser
from .utils.normaliser import vital_normaliser
from .utils.normaliser import lab_normaliser

from .utils.build_vocab import build_vocab

from transformers import AutoTokenizer
from .utils.med_tokenizer import MedTokenizer
from .utils.vitals_tokenizer import VitalTokenizer
from .utils.labs_tokenizer import LabTokenizer

"""
For Multimodal Data
"""
class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, listfile, task_type='phenotype', modalities = ['vital','lab','medicine','text'],
                 lmdb_path_vital = '', lmdb_path_lab = '',
                 lmdb_path_text = '', lmdb_path_medicine = '',
                 text_model_name = 'nlpie/tiny-clinicalbert', text_max_len = 512,
                 batch_size=64, num_workers=4):
        
        super().__init__()
        self.listfile = listfile
        self.modalities = modalities
       
        self.lmdb_path_vital = lmdb_path_vital
        self.lmdb_path_lab = lmdb_path_lab
        self.lmdb_path_text = lmdb_path_text
        self.lmdb_path_medicine = lmdb_path_medicine


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

        vital_categoricals_path = 'mimic_data\data_module.py'
        with open(vital_categoricals_path, 'r') as f:
            vital_categoricals = json.load(f)

        discrete_labels = vital_categoricals.keys()
        self.vitalnorm = vital_normaliser(self.listfile, self.lmdb_path_vital, discrete_labels)
        self.vitals_tokenizer = VitalTokenizer(label_vocab=build_vocab(self.lmdb_path_vital, 'label'), 
                                             vitalnorm=self.vitalnorm, discrete_label_categorical_values=vital_categoricals)

        self.labnorm = lab_normaliser(self.listfile, self.lmdb_path_lab)
        self.labs_tokenizer = LabTokenizer(label_vocab=build_vocab(self.lmdb_path_lab, 'label'), 
                                           labnorm=self.labnorm)


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Called on every GPU separately; stage can be 'fit' or 'test'
        if stage in (None, 'fit'):
            self.train_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='train',
                                           lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab,
                                           lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine)
           
            self.val_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='val',
                                         lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab,
                                         lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine)

        if stage in (None, 'test'):
            self.test_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='test',
                                          lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab,
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
                                        vitals_tokenizer=self.vitals_tokenizer,
                                        labs_tokenizer=self.labs_tokenizer,
                                        task_type=self.task_type, text_max_len=self.text_max_len)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=MultimodalCollate(modalities=self.modalities,
                                        text_tokenizer=self.text_tokenizer,
                                        med_tokenizer=self.med_tokenizer,
                                        vitals_tokenizer=self.vitals_tokenizer,
                                        labs_tokenizer=self.labs_tokenizer,
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
                                         vitals_tokenizer=self.vitals_tokenizer,
                                         labs_tokenizer=self.labs_tokenizer,
                                         task_type=self.task_type, text_max_len=self.text_max_len)
        )
