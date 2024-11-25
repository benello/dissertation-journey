import argparse
import logging

from jedi.api.file_name import complete_file_name

from src.dimension_analysis import DimensionalityAnalyser
from src.model import FeatureVisualizerCNN
from src.novel_generator import NovelGenerator
from src.trainer import ModelTrainer
from src.activation_visualizer import ActivationVisualizer
from src.utils import setup_logging, load_config, create_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Novel MNIST Generator and CNN Feature Visualization')

    # Configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')

    # Data Generation
    parser.add_argument('--use-novel', action='store_true',
                        help='Generate novel MNIST-like dataset')
    parser.add_argument('--representation', type=str, choices=['chinese', 'roman', 'dots', 'all'],
                        default='all', help='Type of number representation to generate')
    parser.add_argument('--samples', type=int,
                        help='Number of samples per class to generate (overrides config)')

    # Training and Model Options
    parser.add_argument('--train', action='store_true',
                        help='Train the model')

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
    parser.add_argument('--font-path', type=str,
                        help='Custom font path (overrides config)')

    args = parser.parse_args()

    # Validation
    if args.digit is not None and (args.digit < 0 or args.digit > 8):
        parser.error("--digit must be between 0 and 8")

    if args.visualize_activations and not (args.train or args.digit is not None):
        parser.error("--visualize-activations requires either --train or --digit")

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

    # Train if requested
    if args.train:
        logger.info("Starting training phase...")
        trainer.train()
        trainer.evaluate()

        # Save trained model
        model_path = output_dir / 'model.pth'
        trainer.save_model(model_path)

    # Evaluate with novel data if requested
    if args.use_novel:
        novel_generator = NovelGenerator(config)
        novel_generator.save_example_images()
        novel_generator.save_dataset_to_mnist()
        trainer.evaluate_novel(novel_generator.output_dir)

    # Visualize if requested
    if args.visualize_dimension_analysis:
        logger.info("Creating feature visualizations...")
        dimension_analyser = DimensionalityAnalyser(model, trainer.train_loader, config)
        dimension_analyser.collect_features()
        fig_pca, fig_analysis = dimension_analyser.visualize()

        # Save pca visualization
        fig_path = output_dir / 'figures' / 'feature_visualization.png'
        fig_pca.savefig(fig_path)
        logger.info(f"Feature visualization saved to {fig_path}")

        # Save dimensionality analysis
        analysis_path = output_dir / 'figures' / 'dimensionality_analysis.png'
        fig_analysis.savefig(analysis_path)
        logger.info(f"Dimensionality analysis saved to {analysis_path}")

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
        fig_evolution = activation_vis.visualize_feature_evolution(sample_image, digit_label)
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