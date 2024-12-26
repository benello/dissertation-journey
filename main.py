import argparse
import logging

import torch
from matplotlib import pyplot as plt
from torch import nn

from src.activation_tracking import ActivationTracker
from src.dimension_analysis import DimensionalityAnalyser
from src.model import FeatureVisualizerCNN, model_name
from src.novel_generator import NovelGenerator
from src.novel_loader import NovelDataset
from src.trainer import ModelTrainer
from src.activation_visualizer import ActivationVisualizer
from src.utils import setup_logging, load_config, create_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Novel MNIST Generator and CNN Feature Visualization')

    # Configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')

    # Data Generation
    parser.add_argument('--novel', action='store_true',
                        help='Generate novel MNIST-like dataset to be used to visualize activations')

    # Model Options
    parser.add_argument('--train', action='store_true',
                        help='Train the model regardless of existing model')

    # Visualization Options
    parser.add_argument('--visualize-dimension-analysis', action='store_true',
                        help='Create feature visualizations')
    parser.add_argument('--visualize-activations', action='store_true',
                        help='Visualize layer activations')
    parser.add_argument('--visualize-examples', action='store_true',
                        help='Generate and save example images for each representation')
    parser.add_argument('--digit', type=int, default=None,
                        help='Specific digit to visualize (0-8)')

    # Advanced Options
    parser.add_argument('--save-mnist', action='store_true',
                        help='Save generated data in MNIST binary format')

    args = parser.parse_args()

    # Validation
    if args.digit is not None and (args.digit < 0 or args.digit > 8):
        parser.error("--digit must be between 0 and 8")

    return args


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
    model_path = output_dir / model_name

    # Train if requested
    if args.train:
        logger.info("Starting training phase...")
        trainer.train()

        # Save the activations during evaluation to disk
        tracker = ActivationTracker(output_dir)
        tracker.set_layers_to_track([nn.Conv2d, nn.ReLU])
        with tracker.track(model, 'evaluation'):
            trainer.evaluate()

        # Save trained model
        trainer.save_model(model_path)
    else:
        logger.info("Loading trained model...")
        trainer.load_model(model_path)

    # Visualize if requested
    if args.visualize_dimension_analysis:
        logger.info("Creating feature visualizations...")
        dimension_analyser = DimensionalityAnalyser(model, trainer.train_loader, config)
        results, fig_analysis = dimension_analyser.run_analysis()

        # Save pca visualization
        analysis_path = output_dir / 'figures' / 'dimensionality_analysis.png'
        fig_analysis.savefig(analysis_path)
        fig_analysis.clear()
        logger.info(f"Dimensionality analysis saved to {analysis_path}")

        logger.info(results)

    # Visualize activations if requested
    if args.visualize_activations:
        logger.info("Creating activation visualizations...")

        # Create activation visualizations
        activation_vis = ActivationVisualizer(model)

        # Evaluate with novel data if requested
        if args.novel:
            novel_generator = NovelGenerator(config)
            novel_generator.save_example_images()
            novel_generator.save_dataset_to_mnist()

            novel_loader = NovelDataset(config['data']['data_dir'] + '/generated_novel')
            cnt = 0
            for image, label in novel_loader:
                novel_fig = activation_vis.visualize_feature_evolution(image, label)
                novel_path = output_dir / 'figures' / 'novel_activations' / f'feature_evolution_digit_{cnt}.png'
                novel_fig.savefig(novel_path)
                plt.close(novel_fig)
                cnt += 1

            logger.info(f"Feature evolution visualization saved to {output_dir / 'figures' / 'novel_activations'}")

        # Get a sample image
        if args.digit is not None:
            sample_image = None
            digit_label = None

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

        # Visualize feature evolution
        fig_evolution = activation_vis.visualize_feature_evolution(sample_image, digit_label)
        analysis_path = output_dir / 'figures' / f'feature_evolution_digit_{digit_label}.png'
        fig_evolution.savefig(analysis_path)
        logger.info(f"Feature evolution visualization saved to {analysis_path}")

        # Get and print most activated channels
        top_channels = activation_vis.get_most_activated_channels(sample_image)
        logger.info("\nMost activated channels per layer:")
        for layer_name, channels in top_channels.items():
            logger.info(f"\n{layer_name}:")
            for idx, (channel, activation) in enumerate(channels, 1):
                logger.info(f"  {idx}. Channel {channel}: {activation:.4f}")


if __name__ == '__main__':
    main()