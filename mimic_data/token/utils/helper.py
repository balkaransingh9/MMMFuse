import torch
def compute_last_seen(label_ids: list[int], times: list[float]) -> torch.Tensor:
    last_seen = {}
    deltas = []
    for i, label in enumerate(label_ids):
        curr_time = times[i]
        if label in last_seen:
            deltas.append(curr_time - last_seen[label])
        else:
            deltas.append(0.0)
        last_seen[label] = curr_time
    return torch.tensor(deltas)