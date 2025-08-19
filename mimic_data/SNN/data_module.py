import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import json 
from pathlib import Path

#from .multimodal_data import MultimodalData
from .multimodal_data_missing import MultimodalData

from .collate.multimodal_collate import MultimodalCollate

from .utils.normaliser import med_normaliser
from .utils.normaliser import vital_normaliser
from .utils.normaliser import value_hour_normaliser
from .utils.normaliser import procedure_normaliser
from .utils.normaliser import normaliser

from .utils.build_vocab import build_vocab

from transformers import AutoTokenizer
from .utils.med_tokenizer import MedTokenizer
from .utils.vitals_tokenizer import VitalTokenizer
from .utils.labs_tokenizer import LabTokenizer
from .utils.procedure_tokenizer import ProcedureTokenizer
from .utils.output_tokenizer import OutputTokenizer

"""
For Multimodal Data
"""
class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, listfile, task_type='phenotype', modalities = ['vital','lab','medicine','text'],
                 lmdb_path_vital = '', lmdb_path_lab = '',
                 lmdb_path_text = '', lmdb_path_medicine = '',
                 lmdb_path_procedure = '', lmdb_path_output = '',
                 lmdb_path_ecg = '',
                 csv_path_demographic = '', csv_path_icd_code = '',
                 text_model_name = 'nlpie/tiny-clinicalbert', 
                 text_max_len = 512, batch_size=64, num_workers=4):
        
        super().__init__()
        self.listfile = listfile
        self.modalities = modalities
       
        self.lmdb_path_vital = lmdb_path_vital
        self.lmdb_path_lab = lmdb_path_lab
        self.lmdb_path_text = lmdb_path_text
        self.lmdb_path_medicine = lmdb_path_medicine
        self.lmdb_path_procedure = lmdb_path_procedure
        self.lmdb_path_output = lmdb_path_output
        self.lmdb_path_ecg = lmdb_path_ecg

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

        #medicine
        self.mednorm = med_normaliser(self.listfile, self.lmdb_path_medicine)
        self.med_label_vocab = build_vocab(self.lmdb_path_medicine, 'label')
        self.med_unit_vocab = build_vocab(self.lmdb_path_medicine, 'amount_std_uom')
        self.med_cat_vocab = build_vocab(self.lmdb_path_medicine, 'ordercategoryname')

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.med_tokenizer = MedTokenizer(self.med_label_vocab, self.med_unit_vocab,
                                          self.med_cat_vocab, self.mednorm)

        #vital
        current_dir = Path(__file__).parent
        vital_categoricals_path = current_dir / 'vital_categoricals.json'
        with open(vital_categoricals_path, 'r') as f:
            vital_categoricals = json.load(f)


        discrete_labels = vital_categoricals.keys()
        self.vitalnorm = vital_normaliser(self.listfile, self.lmdb_path_vital, discrete_labels)
        self.vital_label_vocab = build_vocab(self.lmdb_path_vital, 'label')
        self.vitals_tokenizer = VitalTokenizer(label_vocab=self.vital_label_vocab, 
                                             vitalnorm=self.vitalnorm)
        
        self.discrete_vital2num_categories = {
            self.vital_label_vocab[name]: len(categories)
            for name, categories in vital_categoricals.items()
        }

        #lab
        self.labnorm = value_hour_normaliser(self.listfile, self.lmdb_path_lab)
        self.lab_label_vocab = build_vocab(self.lmdb_path_lab, 'label')
        self.labs_tokenizer = LabTokenizer(label_vocab=self.lab_label_vocab,
                                           labnorm=self.labnorm)
        

        #procedure
        self.procnorm = procedure_normaliser(self.listfile, self.lmdb_path_procedure)
        self.procedure_label_vocab = build_vocab(self.lmdb_path_procedure, 'label')
        self.procedure_tokenizer = ProcedureTokenizer(label_vocab=self.procedure_label_vocab,
                                                      procnorm=self.procnorm)
        
        #output
        self.output_norm = value_hour_normaliser(self.listfile, self.lmdb_path_output)
        self.output_label_vocab = build_vocab(self.lmdb_path_output, 'label')
        self.output_tokenizer = OutputTokenizer(label_vocab=self.output_label_vocab,
                                                outnorm=self.output_norm)

        #demographics
        demographic = pd.read_csv(csv_path_demographic)
        train_set = self.listfile[self.listfile['original_split'] == 'train']
        train_set = train_set['stay_id'].values
        demo_train = demographic[demographic['stay_id'].isin(train_set)]
        age_mean = demo_train['anchor_age'].mean()
        age_std  = demo_train['anchor_age'].std()
        demographic['anchor_age'] = (demographic['anchor_age'] - age_mean) / age_std
        self.demographic = demographic

        #icd codes
        self.icd_code = pd.read_csv(csv_path_icd_code)

        #ecg
        self.ecg_norm = None
        if self.lmdb_path_ecg != '':
           self.ecg_norm = normaliser(self.listfile, lmdb_path=self.lmdb_path_ecg)
        


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Called on every GPU separately; stage can be 'fit' or 'test'
        if stage in (None, 'fit'):
            self.train_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='train',
                                           demographic_file = self.demographic, icd_code_file=self.icd_code,
                                           lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab,
                                           lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine,
                                           lmdb_path_ecg=self.lmdb_path_ecg, ecg_normaliser = self.ecg_norm,
                                           lmdb_path_procedure=self.lmdb_path_procedure, lmdb_path_output=self.lmdb_path_output)
           
            self.val_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='val',
                                         demographic_file = self.demographic, icd_code_file=self.icd_code,
                                         lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab,
                                         lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine,
                                         lmdb_path_ecg=self.lmdb_path_ecg, ecg_normaliser = self.ecg_norm,
                                         lmdb_path_procedure=self.lmdb_path_procedure, lmdb_path_output=self.lmdb_path_output)

        if stage in (None, 'test'):
            self.test_ds = MultimodalData(self.listfile, modalities=self.modalities, task_type=self.task_type, split='test',
                                          demographic_file = self.demographic, icd_code_file=self.icd_code,
                                          lmdb_path_vital=self.lmdb_path_vital, lmdb_path_lab=self.lmdb_path_lab,
                                          lmdb_path_text=self.lmdb_path_text, lmdb_path_medicine=self.lmdb_path_medicine,
                                          lmdb_path_ecg=self.lmdb_path_ecg, ecg_normaliser = self.ecg_norm,
                                          lmdb_path_procedure=self.lmdb_path_procedure, lmdb_path_output=self.lmdb_path_output)

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
                                        procedure_tokenizer=self.procedure_tokenizer,
                                        output_tokenizer=self.output_tokenizer,
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
                                        procedure_tokenizer=self.procedure_tokenizer,
                                        output_tokenizer=self.output_tokenizer,
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
                                         procedure_tokenizer=self.procedure_tokenizer,
                                         output_tokenizer=self.output_tokenizer,
                                         task_type=self.task_type, text_max_len=self.text_max_len)
        )
