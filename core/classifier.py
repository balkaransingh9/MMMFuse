import torch.nn as nn

class SimpleClassifier(nn.Module):
  def __init__(self, in_dim, out_dim, encoder):
    super(SimpleClassifier, self).__init__()
    self.encoder = encoder
    self.classifier = nn.Sequential(
        nn.Linear(in_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
    )
    self.final_layer = nn.Linear(512, out_dim)

  def forward(self, x):
    x = self.encoder(x)
    x = x[:, 0, :]
    x = self.classifier(x)
    x = self.final_layer(x)
    return x