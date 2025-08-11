import torch
from .helper import compute_last_seen

class ProcedureTokenizer:
    def __init__(self, label_vocab, procnorm):
        self.label_vocab = label_vocab
        self.procnorm = procnorm

    def tokenize(self, proc):
        B = len(proc)
        L = max((len(x['label']) for x in proc if x is not None), default=0)
        proc_label_pad = torch.zeros(B, L).long()
        proc_duration_pad = torch.zeros(B, L)
        proc_hours_pad = torch.zeros(B, L)
        proc_hours_norm_pad = torch.zeros(B, L)
        proc_last_seen_pad = torch.zeros(B, L)
        proc_mask = torch.ones(B, L, dtype=torch.bool)

        for i, m in enumerate(proc):
            if m is not None:
                labels = m['label']
                durations = m['procedure_duration']
                hours = m['hours_from_intime']
                l = len(labels)

                label_ids = [self.label_vocab.get(lbl, 0) for lbl in labels]

                proc_label_pad[i, :l] = torch.tensor(label_ids)
                proc_duration_pad[i, :l] = (torch.tensor(durations) - self.procnorm['duration']['mean']) / self.procnorm['duration']['std']
                proc_hours_pad[i, :l] = torch.tensor(hours)
                proc_hours_norm_pad[i, :l] = (proc_hours_pad[i, :l] - self.procnorm['hours']['mean']) / self.procnorm['hours']['std']
                proc_last_seen_pad[i, :l] = compute_last_seen(label_ids, hours)
                proc_mask[i, :l] = False

        return {
            'hours': proc_hours_pad,
            'hours_norm': proc_hours_norm_pad,
            'duration': proc_duration_pad,
            'label': proc_label_pad,
            'last_seen': proc_last_seen_pad,
            'mask': proc_mask
        }