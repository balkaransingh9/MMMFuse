import torch

class ProcedureTokenizer:
    def __init__(self, label_vocab, procnorm):
        self.label_vocab = label_vocab
        self.procnorm = procnorm

    def tokenize(self, proc):
        B = len(proc)
        L = max(len(x['label']) for x in proc if x is not None)
        proc_label_pad = torch.zeros(B, L).long()
        proc_duration_pad = torch.zeros(B, L)
        proc_hours_pad = torch.zeros(B, L)
        proc_hours_norm_pad = torch.zeros(B, L)
        proc_mask = torch.zeros(B, L, dtype=torch.bool)

        for i, m in enumerate(proc):
            if m is not None:
                l = len(m['label'])
                proc_label_pad[i, :l] = torch.tensor([self.label_vocab.get(i) for i in m['label']])
                proc_duration_pad[i, :l] = torch.tensor((m['procedure_duration'] - self.procnorm['duration']['mean']) / self.procnorm['duration']['std'])
                proc_hours_norm_pad[i, :l] = torch.tensor((m['hours_from_intime'] - self.procnorm['hours']['mean']) / self.procnorm['hours']['std'])
                proc_hours_pad[i, :l] = torch.tensor(m['hours_from_intime'])
                proc_mask[i, l:] = True
            else:
                proc_mask[i, :] = True

        return {
            'hours': proc_hours_pad,
            'hours_norm': proc_hours_norm_pad,
            'duration': proc_duration_pad,
            'label': proc_label_pad,
            'mask': proc_mask
        }