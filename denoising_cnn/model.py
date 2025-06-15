import torch.nn as nn

class DenoisingCNN(nn.Module):
    def __init__(self, in_channels=1, dropout_rate=0.5):
        """
        Modelo CNN para redução de ruído.
        Args:
            in_channels (int): 1 para cinza, 3 para RGB, etc.
            dropout_rate (float): taxa de dropout.
        """
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ) for _ in range(7)]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, canais, H, W)
        retorna: (batch, canais, H, W)
        """
        x = self.relu(self.conv1(x))
        x = self.conv_layers(x)
        x = self.conv2(x)
        return x
