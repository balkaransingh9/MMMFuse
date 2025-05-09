import torch.nn as nn

class SimpleClassifier(nn.Module):
  def __init__(self, encoder, encoder_dim, out_dim):
    super(SimpleClassifier, self).__init__()
    self.encoder = encoder
    self.classifier = nn.Sequential(
        nn.Linear(encoder_dim, 512),
        nn.ReLU(),
    )
    self.final_layer = nn.Linear(512, out_dim)

  def forward(self, x, mask):
    x = self.encoder(x, mask)
    x = x[:, 0, :]
    x = self.classifier(x)
    x = self.final_layer(x)
    return x