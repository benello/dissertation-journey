# CNN Feature Visualization

This project provides tools for training a CNN on the MNIST dataset and visualizing the learned features using dimensionality reduction techniques (PCA and t-SNE).

## Features

- Clean, modular implementation of a CNN for MNIST classification
- Feature extraction and visualization using PCA and t-SNE
- Configurable model architecture and training parameters
- Comprehensive visualizations of the feature space

## Project Structure

```
cnn_feature_vis/
├── requirements.txt    # Project dependencies
├── README.md          # This file
├── src/
│   ├── model.py       			# CNN model definition
│   ├── trainer.py     			# Training functionality
│   ├── visualizer.py  			# PCA and visualization code
│   ├── utils.py       			# Helper functions
│   └── activation_visualizer.py	# 
├── config/
│   └── config.yaml    # Configuration parameters
└── main.py           # Main script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/benello/dissertation-journey.git
cd cnn_feature_vis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python main.py --train
```

2. Create visualizations:
```bash
python main.py --visualize
```

3. Train and visualize in one go:
```bash
python main.py --train --visualize
```

4. Visualize activations for a random digit
```bash
python main.py --visualize-activations
```

5. Visualize activations for a specific digit (e.g., 7)
```bash
python main.py --visualize-activations --digit 7
```



## Configuration

The model and training parameters can be configured in `config/config.yaml`. Key parameters include:

- Model architecture (number of layers, channels)
- Training parameters (learning rate, batch size, epochs)
- Visualization settings (number of samples, figure size)

## Visualization Output

The script generates three visualizations:

1. PCA explained variance ratio plot
2. First two PCA components scatter plot
3. t-SNE visualization of the feature space

Output files are saved in the `outputs/figures` directory.