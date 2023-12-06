"""Define the Model:"""

#  imports for the network
import torch.nn as nn
import torch.nn.functional as functional


class FLAME_SCOUT_Model(nn.Module):
    def __init__(self, num_classes):
        super(FLAME_SCOUT_Model, self).__init__()

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Separable Convolution Blocks with Residual Connections
        self.conv_blocks = self._make_conv_blocks(8)

        # 1x1 Convolution for Residual Connection
        self.conv_residual = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)  # Adjusted stride

        # Final Convolutional Block
        self.final_conv = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(8)

        # Global Average Pooling and Output Layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(8, num_classes)

    def _make_conv_blocks(self, size):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=3, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=3, padding=1),
            nn.BatchNorm2d(size),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        previous_block_activation = x

        for block in self.conv_blocks:
            x = block(x)
            # Use 1x1 convolution for residual connection
            residual = self.conv_residual(previous_block_activation)
            residual = functional.interpolate(residual, size=x.size()[2:],
                                              mode='nearest')  # Adjusted to match the spatial dimensions
            x = x + residual
            previous_block_activation = x

        x = functional.relu(self.bn_final(self.final_conv(x)))

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
