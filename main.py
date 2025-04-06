import argparse
import logging

from matplotlib import pyplot as plt

from src.ResultsHelper import export_csv_data
from src.activation_analysis import ActivationAnalysis
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
    parser.add_argument('--sample-index', type=int, default=None,
                        help='Specific sample index to visualize from the dataset')

    # Advanced Options
    parser.add_argument('--save-mnist', action='store_true',
                        help='Save generated data in MNIST binary format')

    args = parser.parse_args()

    # Validation
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample-index must be a non-negative integer")

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
        trainer.evaluate()

        # Save trained model
        trainer.save_model(model_path)
    else:
        logger.info("Loading trained model...")
        trainer.load_model(model_path)

    novel_generator = NovelGenerator(config)
    novel_generator.save_example_images()
    novel_generator.save_dataset_to_mnist()

    if args.sample_index is not None:
        # Get the specific sample by index
        dataset = trainer.test_loader.dataset
        if 0 <= args.sample_index < len(dataset):
            sample_image, digit_label = dataset[args.sample_index]
            logger.info(f"Selected sample at index {args.sample_index} with label {digit_label}")
        else:
            logger.error(f"Sample index {args.sample_index} out of range (0-{len(dataset) - 1})")
            return
    else:
        sample_image, digit_label = next(iter(trainer.train_loader))
        sample_image = sample_image[0]
        digit_label = digit_label[0].item()

    sample_image = sample_image.unsqueeze(0)  # Add batch dimension

    novel_loader = NovelDataset(config['data']['data_dir'] + '/generated_novel')

    dimension_analyser = DimensionalityAnalyser(model, config)
    tracked_test_features, test_labels = dimension_analyser.collect_features(trainer.test_loader, 'mnist', config)

    # Visualize if requested
    if args.visualize_dimension_analysis:
        logger.info("Creating feature visualizations...")

        results, fig_analysis = dimension_analyser.run_analysis(tracked_test_features)

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
        activation_vis = ActivationVisualizer(model, config)

        # Evaluate with novel data if requested
        if args.novel:
            cnt = 0
            for image, label in novel_loader:
                novel_fig = activation_vis.visualize_feature_evolution(image, label)
                novel_path = output_dir / 'figures' / 'novel_activations' / f'feature_evolution_digit_{cnt}.png'
                novel_fig.savefig(novel_path)
                plt.close(novel_fig)
                cnt += 1

            logger.info(f"Feature evolution visualization saved to {output_dir / 'figures' / 'novel_activations'}")

        # Visualize feature evolution
        fig_evolution = activation_vis.visualize_feature_evolution(sample_image, digit_label)
        inactive_fig = activation_vis.visualize_inactive_channels(sample_image)
        inactive_fig.savefig(output_dir / 'figures' / 'empty_activations.png')
        inactive_fig.clear()
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

    logger.info('Calculating feature distances')
    activation_analyser = ActivationAnalysis(model, config)
    pca_models = activation_analyser.perform_pca_analysis(tracked_test_features, test_labels)

    activation_analyser.analyse_samples(pca_models, test_labels, novel_loader)

    if args.sample_index is not None:
        activation_analyser.analyse_sample(pca_models, test_labels, sample_image, 'Normal')
    else:
        activation_analyser.analyse_samples(pca_models, test_labels, trainer.test_loader)

    #if args.export_result:
    export_csv_data(trainer, model, config, pca_models, test_labels, output_dir)

if __name__ == '__main__':
    main()