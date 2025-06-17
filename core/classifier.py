import torch.nn as nn

class SimpleClassifier(nn.Module):
  def __init__(self, encoder, encoder_dim, num_classes):
    super(SimpleClassifier, self).__init__()
    self.encoder = encoder

    self.classifier = nn.Sequential(
      nn.Linear(encoder_dim, encoder_dim),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(encoder_dim, num_classes)
      )

  def forward(self, input):
    x = self.encoder(x, input)
    x = self.classifier(x)
    return x