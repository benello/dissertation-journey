Channels are like feature detectors:
- One channel might detect edges
- Another might detect corners
- Another might detect textures

Neurons are like feature checkers:
- Each neuron checks for its feature
- At a specific location
- Using the channel's shared pattern


# MNIST Dataset

## Files
```
train-images-idx3-ubyte.gz  (Training images - 60,000)
train-labels-idx1-ubyte.gz  (Training labels - 60,000)
t10k-images-idx3-ubyte.gz   (Test images - 10,000)
t10k-labels-idx1-ubyte.gz   (Test labels - 10,000)
```

## File Structure
### Image Files
1. Magic number: (first 4 bytes)
2. Number of images (4 bytes)
3. Number of rows: 28 (4 bytes)
4. Number of columns: 28 (4 bytes)
5. Image data (unsigned bytes)

### Label Files
1. Magic number: 2049 (first 4 bytes)
2. Number of labels (4 bytes)
3. Label data (unsigned bytes)

## Data Format
- Images: 28x28 grayscale pixels, values 0-255
- Labels: Single digits 0-9
- All integers stored in big-endian format
- Data stored as raw bytes after header