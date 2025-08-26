import torch

class MultimodalCollate:
    def __init__(self, modalities, task_type='mortality'):
        if task_type not in ['mortality', 'readmission']:
            raise ValueError(f"Unsupported task type: {task_type}")
        self.modalities = modalities
        self.task_type  = task_type

    @staticmethod
    def _to_f32(x):
        if torch.is_tensor(x):
            return x.float() if x.dtype.is_floating_point else x
        if isinstance(x, dict):
            return {k: MultimodalCollate._to_f32(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            vals = [MultimodalCollate._to_f32(v) for v in x]
            return type(x)(vals) if isinstance(x, tuple) else vals
        return x

    @staticmethod
    def _prep_seq_modality(items, drop_first_col=False):
        """
        items: list of per-sample payloads for a modality.
               Each item can be:
               - dict with 'value' (TxD) and optional 'mask' (TxD)
               - tensor/ndarray (TxD)
               - None (missing)
        Returns: list of dicts with 'value' (float32, NaNs->0, optional col-drop) and 'mask' (float32).
        """
        # Find a reference shape
        ref_val = None
        for it in items:
            if it is None:
                continue
            if isinstance(it, dict) and 'value' in it and it['value'] is not None:
                ref_val = torch.as_tensor(it['value'])
                break
            elif torch.is_tensor(it) or (it is not None):
                try:
                    ref_val = torch.as_tensor(it)
                    break
                except Exception:
                    pass

        out = []
        for it in items:
            if it is None:
                if ref_val is None:
                    out.append(None)
                else:
                    zeros = torch.zeros_like(ref_val, dtype=torch.float32)
                    mask  = torch.zeros_like(ref_val, dtype=torch.float32)
                    if drop_first_col:
                        zeros = zeros[..., 1:]
                        mask  = mask[..., 1:]
                    out.append({'value': zeros, 'mask': mask})
                continue

            # Normalize shape/content
            if isinstance(it, dict) and 'value' in it:
                val = torch.as_tensor(it['value'], dtype=torch.float32)
                mask = torch.as_tensor(it.get('mask', torch.isfinite(val).float()), dtype=torch.float32)
            else:
                val = torch.as_tensor(it, dtype=torch.float32)
                mask = torch.isfinite(val).float()

            # Drop the first column (hour) if requested
            if drop_first_col and val.ndim >= 2 and val.shape[1] > 1:
                val  = val[:, 1:]
                mask = mask[:, 1:]

            # Replace NaNs with 0 and zero-out mask
            is_finite = torch.isfinite(val)
            mask = mask * is_finite.float()
            val = torch.where(is_finite, val, torch.zeros_like(val))

            out.append({'value': val, 'mask': mask})

        return out

    def __call__(self, batch):
        # batch: iterable of (outs, flags, label)
        outs_list, flags_list, labels_list = zip(*batch)

        seq_data = {}

        # Static modalities
        if 'demographic' in self.modalities:
            demo = [o['demographic'] for o in outs_list]
            seq_data['demographic'] = torch.stack(demo, dim=0)

        if 'diagnosis' in self.modalities:
            icd = [o['diagnosis'] for o in outs_list]
            seq_data['diagnosis'] = torch.stack(icd, dim=0)

        if 'treatment' in self.modalities:
            trt = [o['treatment'] for o in outs_list]
            seq_data['treatment'] = torch.stack(trt, dim=0)

        # Time-series modalities (remove hour col)
        if 'vital' in self.modalities:
            vital_items = [o['vital'] for o in outs_list]
            seq_data['vital'] = self._prep_seq_modality(vital_items, drop_first_col=True)

        if 'lab' in self.modalities:
            lab_items = [o['lab'] for o in outs_list]
            seq_data['lab'] = self._prep_seq_modality(lab_items, drop_first_col=True)

        if 'medicine' in self.modalities:
            meds = [o['medicine'] for o in outs_list]
            seq_data['medicine'] = meds

        # Present mask (sample-level, modality present or missing)
        present_mask = {}
        for mod in self.modalities:
            mask = torch.tensor(
                [not f.get(f"{mod}_missing", True) for f in flags_list],
                dtype=torch.bool
            ).float()
            present_mask[mod] = mask

        labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
        seq_data = self._to_f32(seq_data)

        return {
            "inputs": seq_data,
            "present_mask": present_mask,
            "labels": labels
        }