import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model, config):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = model.device
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )
        
        self.train_loader, self.test_loader = self._get_data_loaders()
        
    def _get_data_loaders(self):
        """Create train and test data loaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.config['data']['train_normalize_mean'],
                self.config['data']['train_normalize_std']
            )
        ])
        
        train_dataset = datasets.MNIST(
            self.config['data']['data_dir'], 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            self.config['data']['data_dir'], 
            train=False, 
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        self.model.train()
        
        for epoch in range(self.config['training']['epochs']):
            total_loss = 0
            for (data, target) in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            
            avg_loss = total_loss / len(self.train_loader)
            logger.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}, '
                       f'Average Loss: {avg_loss:.4f}')
    
    def save_model(self, path):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self):
        """Evaluate the model on test data."""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        logger.info(f'Test set: Average loss: {test_loss:.4f}, '
                   f'Accuracy: {correct}/{len(self.test_loader.dataset)} '
                   f'({accuracy:.2f}%)')