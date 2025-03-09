from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
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
        self.config = config['training']
        self.device = model.device
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )

        self.train_metrics = {'loss': [], 'accuracy': []}
        self.val_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        self.train_loader, self.test_loader = self._get_data_loaders(config['data'])
        
    def _get_data_loaders(self, data_config):
        """Create train and test data loaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                data_config['train_normalize_mean'],
                data_config['train_normalize_std'],
            )
        ])
        
        train_dataset = datasets.MNIST(
            data_config['data_dir'],
            train=True,
            download=True, 
            transform=transform,
        )
        
        test_dataset = datasets.MNIST(
            data_config['data_dir'],
            train=False, 
            transform=transform,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
        )
        
        return train_loader, test_loader

    def train(self):
        """Train with aggregated metrics per epoch."""
        logger.info("Starting training...")
        self.model.train()

        for epoch in range(self.config['epochs']):
            epoch_preds = []
            epoch_targets = []
            total_loss = 0

            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_preds.extend(pred.cpu().numpy())
                epoch_targets.extend(target.cpu().numpy())

            # Calculate aggregated metrics
            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100. * np.mean(np.array(epoch_preds) == np.array(epoch_targets))
            precision, recall, f1, _ = precision_recall_fscore_support(
                epoch_targets, epoch_preds, average='weighted', zero_division=0
            )

            logger.info(
                f'Epoch {epoch + 1}/{self.config["epochs"]} metrics:\n'
                f'Loss: {avg_loss:.4f} | '
                f'Accuracy: {accuracy:.2f}% | '
                f'F1: {f1:.4f} | '
                f'Precision: {precision:.4f} | '
                f'Recall: {recall:.4f}'
            )
    
    def save_model(self, path):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist")

        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

    def evaluate(self):
        """Evaluate the model on test data."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_batches = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                loss, corr, prec, rec, f1 = self._eval_core(data, target)
                test_loss += loss
                correct += corr
                total_precision += prec
                total_recall += rec
                total_f1 += f1
                num_batches += 1

        # Average the metrics
        test_loss /= num_batches
        accuracy = 100. * correct / len(self.test_loader.dataset)
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        avg_f1 = total_f1 / num_batches

        logger.info(
            f'Test set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%), '
            f'Precision: {avg_precision:.4f}, '
            f'Recall: {avg_recall:.4f}, '
            f'F1: {avg_f1:.4f}'
        )

        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }

    def _eval_core(self, data, target):
        """Evaluate a single batch."""
        data, target = data.to(self.device), target.to(self.device)
        output, _ = self.model(data)
        test_loss = self.criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        # Calculate batch metrics
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_np, pred_np, average='weighted', zero_division=0
        )

        return test_loss, correct, precision, recall, f1