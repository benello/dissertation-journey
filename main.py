import argparse
from pathlib import Path
import logging

from src.model import FeatureVisualizerCNN
from src.trainer import ModelTrainer
from src.visualizer import FeatureVisualizer
from src.activation_visualizer import ActivationVisualizer
from src.utils import setup_logging, load_config, create_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='CNN Feature Visualization')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--visualize', action='store_true',
                        help='Create feature visualizations')
    parser.add_argument('--visualize-activations', action='store_true',
                        help='Visualize layer activations')
    parser.add_argument('--digit', type=int, default=None,
                        help='Specific digit to visualize activations for')
    return parser.parse_args()


def main():
    # Setup
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    output_dir = create_output_dirs()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Create model
    model = FeatureVisualizerCNN(
        conv_channels=config['model']['conv_channels'],
        kernel_size=config['model']['kernel_size'],
        num_classes=config['model']['num_classes'],
        config=config,
    )

    # Initialize trainer
    trainer = ModelTrainer(model, config)

    # Train if requested
    if args.train:
        logger.info("Starting training phase...")
        trainer.train()
        trainer.evaluate()

        # Save trained model
        model_path = output_dir / 'model.pth'
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    # Visualize if requested
    if args.visualize:
        logger.info("Creating feature visualizations...")
        visualizer = FeatureVisualizer(model, trainer.train_loader, config)
        visualizer.collect_features()
        fig = visualizer.visualize()

        # Save figure
        fig_path = output_dir / 'figures' / 'feature_visualization.png'
        fig.savefig(fig_path)
        logger.info(f"Visualization saved to {fig_path}")

    # Visualize activations if requested
    if args.visualize_activations:
        logger.info("Creating activation visualizations...")

        # Get a sample image
        if args.digit is not None:
            # Find an image of the requested digit
            for images, labels in trainer.train_loader:
                digit_idx = (labels == args.digit).nonzero(as_tuple=True)[0]
                if len(digit_idx) > 0:
                    sample_image = images[digit_idx[0]]
                    digit_label = args.digit
                    break
        else:
            # Get first image from loader
            sample_image, digit_label = next(iter(trainer.train_loader))
            sample_image = sample_image[0]
            digit_label = digit_label[0].item()

        # Create activation visualizations
        activation_vis = ActivationVisualizer(model)

        # Visualize feature evolution
        fig_evolution = activation_vis.visualize_feature_evolution(
            sample_image, digit_label)
        fig_path = output_dir / 'figures' / f'feature_evolution_digit_{digit_label}.png'
        fig_evolution.savefig(fig_path)
        logger.info(f"Feature evolution visualization saved to {fig_path}")

        # Get and print most activated channels
        top_channels = activation_vis.get_most_activated_channels(sample_image)
        logger.info("\nMost activated channels per layer:")
        for layer_name, channels in top_channels.items():
            logger.info(f"\n{layer_name}:")
            for idx, (channel, activation) in enumerate(channels, 1):
                logger.info(f"  {idx}. Channel {channel}: {activation:.4f}")


if __name__ == '__main__':
    main()