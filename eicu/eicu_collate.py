import torch

class MultimodalCollate:
    def __init__(
        self,
        modalities,
        task_type='mortality',
    ):
        if task_type not in ['mortality', 'readmission']:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.modalities     = modalities
        self.task_type      = task_type

    # --- helper: convert ONLY floating tensors (fp64/fp32/fp16/bf16) to float32 ---
    def _to_f32(self, obj):
        if torch.is_tensor(obj):
            return obj.float() if obj.dtype.is_floating_point else obj
        if isinstance(obj, dict):
            return {k: self._to_f32(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            items = [self._to_f32(v) for v in obj]
            return type(obj)(items) if isinstance(obj, tuple) else items
        return obj

    def __call__(self, batch):
        # batch: iterable of (outs, flags, label)
        outs_list, flags_list, labels_list = zip(*batch)

        seq_data = {}

        # Demographics
        if 'demographic' in self.modalities:
            demo = [o['demographic'] for o in outs_list]
            seq_data['demographic'] = torch.stack(demo, dim=0)

        # Diagnosis
        if 'diagnosis' in self.modalities:
            icd = [o['diagnosis'] for o in outs_list]
            seq_data['diagnosis'] = torch.stack(icd, dim=0)

        # Treatment
        if 'treatment' in self.modalities:
            icd = [o['treatment'] for o in outs_list]
            seq_data['treatment'] = torch.stack(icd, dim=0)

        # Vitals
        if 'vital' in self.modalities:
            vital = [o['vital'] for o in outs_list]
            # seq_data['vital'] = self.vitals_tok.tokenize(vital, **self.vitals_kwargs)
            seq_data['vital'] = vital

        # Labs
        if 'lab' in self.modalities:
            lab = [o['lab'] for o in outs_list]
            # seq_data['lab'] = self.labs_tok.tokenize(lab, **self.labs_kwargs)
            seq_data['lab'] = lab

        # Medicine
        if 'medicine' in self.modalities:
            meds = [o['medicine'] for o in outs_list]
            # seq_data['medicine'] = self.med_tok.tokenize(meds, **self.med_kwargs)
            seq_data['medicine'] = meds

        present_mask = {}
        for mod in self.modalities:
            mask = torch.tensor(
                [not f.get(f"{mod}_missing", True) for f in flags_list],
                dtype=torch.bool
            )
            present_mask[mod] = mask.float()  # keep as float for losses that expect float masks

        # Labels
        labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
        seq_data = self._to_f32(seq_data)

        return {
            "inputs": seq_data,
            "present_mask": present_mask,
            "labels": labels
        }