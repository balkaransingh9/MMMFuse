import torch
from torch.nn.utils.rnn import pad_sequence
from MMMFuse.mimic_data.tabular.utils.text_batch_tokenizer import process_text_batch_with_mask
from MMMFuse.mimic_data.tabular.utils.text_embed_tokenizer import prepare_embedding_batch

class MultimodalCollate:
    def __init__(
        self,
        modalities,
        vitals_tokenizer,
        labs_tokenizer,
        med_tokenizer,
        text_tokenizer,
        output_tokenizer,
        procedure_tokenizer,
        task_type='phenotype',
        text_max_len=512,
        vitals_kwargs=None,
        labs_kwargs=None,
        text_kwargs=None,
        med_kwargs=None,
        output_kwargs=None,
        procedure_kwargs=None,
    ):
        """
        Args:
            text_tokenizer: HF-style tokenizer for text modality
            med_tokenizer: custom tokenizer for med modality
            modalities: list of modality names, e.g. ['text','medicine','lab','ecg',…]
            task_type: one of ['phenotype', 'in_hospital_mortality', 'length_of_stay']
        """
        if task_type not in ['phenotype', 'in_hospital_mortality', 'length_of_stay']:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.modalities   = modalities
        self.vitals_tok   = vitals_tokenizer
        self.labs_tok     = labs_tokenizer
        self.med_tok      = med_tokenizer
        self.text_tok     = text_tokenizer
        self.output_tok   = output_tokenizer
        self.procedure_tok = procedure_tokenizer
        self.task_type    = task_type
        self.text_max_len = text_max_len
        self.vitals_kwargs = vitals_kwargs or {}
        self.labs_kwargs   = labs_kwargs  or {}
        self.text_kwargs  = text_kwargs or {}
        self.med_kwargs   = med_kwargs  or {}
        self.output_kwargs = output_kwargs or {}
        self.procedure_kwargs = procedure_kwargs or {}

    def __call__(self, batch):
        outs_list, flags_list, labels_list = zip(*batch)

        seq_data = {}
        
        # Demographics
        if 'demographic' in self.modalities:
            demo = [o['demographic'] for o in outs_list]
            seq_data['demographic'] = torch.stack(demo, dim=0)

        # ICD codes
        if 'icd_code' in self.modalities:
            icd = [o['icd_code'] for o in outs_list]
            seq_data['icd_code'] = torch.stack(icd, dim=0)

        # ECG
        if 'ecg' in self.modalities:
            ecg_list = [o['ecg'] for o in outs_list]
            first_non_none = next((e for e in ecg_list if e is not None), None)
            if first_non_none is None:
                dummy_shape = (1, 5000)
                ecg_list = [torch.zeros(dummy_shape, dtype=torch.float32) for _ in ecg_list]
            else:
                dummy_shape = first_non_none.shape
                ecg_list = [e if e is not None else torch.zeros(dummy_shape, dtype=first_non_none.dtype) for e in ecg_list]
            seq_data['ecg'] = torch.stack(ecg_list, dim=0)

        # Vitals
        if 'vital' in self.modalities:
            vital = [o['vital'] for o in outs_list]
            seq_data['vital'] = self.vitals_tok.tokenize(vital, **self.vitals_kwargs)

        # Labs
        if 'lab' in self.modalities:
            lab = [o['lab'] for o in outs_list]
            seq_data['lab'] = self.labs_tok.tokenize(lab, **self.labs_kwargs)

        # Medicine
        if 'medicine' in self.modalities:
            meds = [o['medicine'] for o in outs_list]
            seq_data['medicine'] = self.med_tok.tokenize(meds, **self.med_kwargs)
        
        # Output
        if 'output' in self.modalities:
            output = [o['output'] for o in outs_list]
            seq_data['output'] = self.output_tok.tokenize(output, **self.output_kwargs)

        # Procedure
        if 'procedure' in self.modalities:
            procedure = [o['procedure'] for o in outs_list]
            seq_data['procedure'] = self.procedure_tok.tokenize(procedure, **self.procedure_kwargs)

        # Text
        if 'text' in self.modalities:
            texts = [o['text'] for o in outs_list]
            tokenized = prepare_embedding_batch(texts, num_notes=5)
            seq_data['text'] = tokenized

        # CXR (now single image per sample)
        if 'cxr' in self.modalities:
            imgs_b, hrs_b, mask_b = [], [], []
            # Determine dtype/shape lazily
            first_present = next((o['cxr'] for o in outs_list if (o.get('cxr') is not None)), None)
            if first_present is not None and isinstance(first_present, dict) and 'img' in first_present:
                c, h, w = first_present['img'].shape  # [1,224,224]
                img_dtype = first_present['img'].dtype
            else:
                c, h, w = 1, 224, 224
                img_dtype = torch.float32

            for o, f in zip(outs_list, flags_list):
                missing = f.get('cxr_missing', True)
                if missing or (o.get('cxr') is None):
                    imgs_b.append(torch.zeros((c, h, w), dtype=img_dtype))
                    hrs_b.append(torch.tensor(0.0, dtype=torch.float32))
                    mask_b.append(False)
                else:
                    entry = o['cxr']
                    imgs_b.append(entry['img'])  # [1,224,224]
                    hrs_b.append(entry['hr'].to(torch.float32))
                    mask_b.append(True)

            seq_data['cxr'] = {
                'img':  torch.stack(imgs_b, dim=0),    # [B,1,224,224]
                'hr':   torch.stack(hrs_b, dim=0),     # [B]
                'mask': torch.tensor(mask_b, dtype=torch.bool),  # [B]
            }

        # Missing‐modality mask
        present_mask = {}
        for mod in self.modalities:
            mask = torch.tensor(
                [ not f.get(f"{mod}_missing", True) for f in flags_list ],
                dtype=torch.bool
            )
            present_mask[mod] = mask.float()

        # Labels
        if self.task_type == 'phenotype':
            labels = torch.stack(labels_list, dim=0)
        elif self.task_type == 'in_hospital_mortality':
            labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
        else:  # length_of_stay
            labels = torch.tensor(labels_list).long()

        return {
            "inputs": seq_data,
            "present_mask": present_mask,
            "labels": labels
        }