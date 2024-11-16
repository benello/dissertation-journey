import torch
import torch.nn as nn

class FeatureVisualizerCNN(nn.Module):
    """CNN model that outputs both predictions and intermediate features."""
    
    def __init__(self, conv_channels, kernel_size, num_classes, config):
        """
        Initialize the CNN model.
        
        Args:
            conv_channels (list): List of channel sizes for conv layers
            kernel_size (int): Size of convolutional kernels
            num_classes (int): Number of output classes
        """
        super(FeatureVisualizerCNN, self).__init__()
        
        # Build convolutional layers dynamically
        layers = []
        in_channels = 1  # MNIST has 1 input channel
        
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.ReLU(),
            ])
            in_channels = out_channels
        
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ])

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_channels[-1], num_classes)
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available()
                                   else 'cpu')
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (predictions, features)
        """
        features = self.conv_layers(x)
        output = self.fc(features)
        return output, features
    
    def get_feature_dim(self):
        """Get the dimension of the feature space."""
        return self.fc.in_features