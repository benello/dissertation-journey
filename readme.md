# CNN Feature Visualization

This project provides tools for training a CNN on the MNIST dataset and visualizing the learned features using dimensionality reduction techniques (PCA and t-SNE).

## Features

- Feature extraction and visualization using PCA, t-SNE, Parallel Analysis and Kaiser Harris
- Configurable model architecture and training parameters

## Project Structure

```
dissertation-journey/
├── requirements.txt    # Project dependencies
├── README.md           # This file
├── main.py             # Main script
├── src/
│   ├── model.py                    # CNN model definition
│   ├── trainer.py                  # Training functionality
│   ├── dimension_analysis.py       # Advanced dimensionality analysis using Parallel Analysis, PCA, Kaiser Harris
│   ├── activation_visualizer.py    # Layer activation visualization
│   ├── novel_generator.py          # Novel number representation generator
│   ├── novel_loader.py             # Dataset loader for novel representations
│   └── utils.py                    # Helper functions
├── config/
│   └── config.yaml     # Configuration parameters
└── assets/
    └── fonts/                      # Font files for number generation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/benello/dissertation-journey.git
cd dissertation-journey
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
python main.py --visualize-dimension-analysis
```

3. Train and visualize in one go:
```bash
python main.py --train --visualize-dimension-analysis
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

The script generates the following visualizations:

1. PCA explained variance ratio plot
2. First two PCA components scatter plot
3. t-SNE visualization of the feature space
4. Parallel Analysis dimension analysis
5. Kaiser Harris dimension analysis

Output files are saved in the `outputs/figures` directory.