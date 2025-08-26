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
    def _extract_val_mask(item, drop_first_col: bool):
        """
        Convert a per-sample payload into (val, mask), dropping first feature column if requested.
        - item can be dict with 'value' and optional 'mask', or a raw tensor/ndarray, or None.
        - returns (val, mask) as float32 tensors, or (None, None) if item is None.
        """
        if item is None:
            return None, None

        if isinstance(item, dict) and 'value' in item:
            val = torch.as_tensor(item['value'], dtype=torch.float32)
            mask = item.get('mask', None)
            mask = torch.as_tensor(mask, dtype=torch.float32) if mask is not None else torch.isfinite(val).float()
        else:
            val = torch.as_tensor(item, dtype=torch.float32)
            mask = torch.isfinite(val).float()

        # drop left-most column (hour)
        if drop_first_col and val.ndim >= 2 and val.shape[1] > 0:
            # assume shape T x D
            if val.shape[1] > 1:
                val  = val[:, 1:]
                mask = mask[:, 1:]
            else:
                # if only the hour column exists, drop to shape T x 0
                val  = val[:, 0:0]
                mask = mask[:, 0:0]

        # replace NaN/inf with 0; zero out mask there
        finite = torch.isfinite(val)
        mask = mask * finite.float()
        val = torch.where(finite, val, torch.zeros_like(val))
        return val, mask

    @staticmethod
    def _pad_and_stack(seq_items, drop_first_col: bool):
        """
        seq_items: list of per-sample items (dict/tensor/None).
        Returns dict with:
          'value': (B, T_max, D) float32
          'mask' : (B, T_max, D) float32
        If ALL samples are missing or reduce to D=0, returns empty tensors with proper batch dim.
        """
        vals, masks = [], []
        D = None
        T_max = 0

        # first pass: extract tensors and infer D, T_max
        tmp = []
        for it in seq_items:
            val, mask = MultimodalCollate._extract_val_mask(it, drop_first_col=drop_first_col)
            tmp.append((val, mask))
            if val is not None:
                if D is None:
                    # infer feature dim (handle T x 0 edge-case)
                    D = val.shape[1] if (val.ndim >= 2) else 1
                T_max = max(T_max, val.shape[0])

        if D is None:
            # all missing or all dropped -> return empty feature tensors
            B = len(seq_items)
            empty = torch.zeros((B, 0, 0), dtype=torch.float32)
            return {'value': empty, 'mask': empty}

        # second pass: pad to (T_max, D), fill missing samples with zeros
        for (val, mask) in tmp:
            if val is None:
                # create zeros for missing sample
                v = torch.zeros((T_max, D), dtype=torch.float32)
                m = torch.zeros((T_max, D), dtype=torch.float32)
            else:
                # ensure val has expected D (it can be 0 if only hour was present)
                cur_D = val.shape[1] if val.ndim >= 2 else 1
                if cur_D != D:
                    # pad feature dim if needed (rare; keeps code robust)
                    pad_feat = D - cur_D
                    if pad_feat < 0:
                        val  = val[:, :D]
                        mask = mask[:, :D]
                    else:
                        val  = torch.nn.functional.pad(val, (0, pad_feat))
                        mask = torch.nn.functional.pad(mask, (0, pad_feat))

                # pad time dim
                pad_T = T_max - val.shape[0]
                if pad_T > 0:
                    pad_shape = (pad_T, 0) if val.ndim == 2 else (pad_T,)
                    val  = torch.nn.functional.pad(val, (0, 0, 0, pad_T))
                    mask = torch.nn.functional.pad(mask, (0, 0, 0, pad_T))
                v, m = val, mask

            vals.append(v)
            masks.append(m)

        value = torch.stack(vals, dim=0)  # (B, T_max, D)
        mask  = torch.stack(masks, dim=0) # (B, T_max, D)
        return {'value': value, 'mask': mask}

    def __call__(self, batch):
        # batch: iterable of (outs, flags, label)
        outs_list, flags_list, labels_list = zip(*batch)

        seq_data = {}

        if 'demographic' in self.modalities:
            demo = [o['demographic'] for o in outs_list]
            seq_data['demographic'] = torch.stack(demo, dim=0)

        if 'diagnosis' in self.modalities:
            icd = [o['diagnosis'] for o in outs_list]
            seq_data['diagnosis'] = torch.stack(icd, dim=0)

        if 'treatment' in self.modalities:
            trt = [o['treatment'] for o in outs_list]
            seq_data['treatment'] = torch.stack(trt, dim=0)

        if 'vital' in self.modalities:
            vital_items = [o['vital'] for o in outs_list]
            seq_data['vital'] = self._pad_and_stack(vital_items, drop_first_col=True)

        if 'lab' in self.modalities:
            lab_items = [o['lab'] for o in outs_list]
            seq_data['lab'] = self._pad_and_stack(lab_items, drop_first_col=True)

        if 'medicine' in self.modalities:
            meds = [o['medicine'] for o in outs_list]
            seq_data['medicine'] = self._pad_and_stack(meds, drop_first_col=True)

        # sample-level presence mask
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
            "inputs": seq_data,          # tensors for vitals/labs; dicts: {'value': (B,T,D), 'mask': (B,T,D)}
            "present_mask": present_mask, # (B,) per modality
            "labels": labels
        }