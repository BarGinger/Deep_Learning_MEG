import torch
import torch.nn as nn

class MEGClassifier(nn.Module):
    def __init__(
        self,
        input_channels=248,
        num_temporal_filters=8,
        temporal_kernel_size=64,
        dropout_rate=0.5,
        activation='elu',
        pool_size=4,
        num_classes=4,
        conv2d_channels=[128, 256],
        transformer_d_model=256,
        transformer_nhead=8,
        transformer_dim_feedforward=512,
        transformer_layers=4,
        fc_hidden=128,
        fc_dropout=0.3
    ):
        super(MEGClassifier, self).__init__()

        # 1D CNN for Temporal Feature Extraction
        self.cnn1d = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        # 2D CNN for Spatial Feature Extraction
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Transformer Encoder for Temporal Attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512),
            num_layers=4
        )

        # Fully Connected Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1D CNN
        x = self.cnn1d(x)

        # Reshape for 2D CNN (batch, 1, channels, seq_len)
        x = x.unsqueeze(1)
        x = self.cnn2d(x)

        # Reshape for Transformer (batch, seq_len, feature)
        x = x.flatten(2).permute(2, 0, 1)  # (seq_len, batch, 256)
        x = self.transformer(x)  # (seq_len, batch, 256)

        # Global Average Pooling
        x = x.mean(dim=0)  # (batch, 256)

        # Classification
        x = self.fc(x)

        return x
