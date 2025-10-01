import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1000):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim] or [batch, embed_dim]
        # If x is [batch, seq_len, embed_dim], pool it (e.g., take CLS token or mean)
        if x.dim() == 3:
            # Use mean pooling (or use x[:, 0] for CLS token if available)
            x = x.mean(dim=1)
        return self.linear(x)