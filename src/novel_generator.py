import logging
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import urllib.request
import sys

image_filename = 'novel_images.ubyte'
label_filename = 'novel_labels.ubyte'

logger = logging.getLogger(__name__)


class NovelGenerator:
    """Generate alternative representations of numbers and formats them in the same structure of the MNIST dataset."""

    def __init__(self, config):
        """
        Initialize the generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config['generation']
        self.output_dir = Path(config['data']['data_dir']) / 'generated_novel'
        self.font = self._initialize_font()
        # Adjust font size based on image size
        self.font_size = int(self.config['image_size'] * 0.75)  # Font size proportional to image size

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_font(self):
        """Initialize and load the required font."""
        try:
            font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
            fonts_dir = Path("assets/fonts")
            fonts_dir.mkdir(parents=True, exist_ok=True)
            font_path = fonts_dir / "NotoSansCJKjp-Regular.otf"

            if not font_path.exists():
                logger.info("Downloading font file... This may take a moment.")
                try:
                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if total_size > 0:
                            percent = min(100, downloaded * 100 / total_size)
                            sys.stdout.write(f"\rDownloading font: {percent:.1f}%")
                            sys.stdout.flush()

                    urllib.request.urlretrieve(font_url, font_path, reporthook=report_progress)
                    logger.info("\nFont downloaded successfully!")
                except Exception as e:
                    raise RuntimeError(f"Failed to download font: {e}")

            return ImageFont.truetype(str(font_path), size=self.config['image_size'])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize font: {e}")

    def create_base_image(self):
        """Create a blank image with white background."""
        image_size = self.config['image_size']
        return Image.new('L', (image_size, image_size), 'black')

    def _get_font_size_for_char(self, char, target_size):
        """Binary search to find the optimal font size."""
        min_size = 1
        max_size = target_size * 2
        optimal_size = target_size
        target_ratio = 0.7  # Target character size relative to image size

        while min_size < max_size:
            mid_size = (min_size + max_size) // 2
            self.font.size = mid_size

            # Get character dimensions
            temp_img = Image.new('L', (target_size, target_size))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), char, font=self.font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Calculate how much of the image the character fills
            size_ratio = max(width, height) / target_size

            if abs(size_ratio - target_ratio) < 0.05:
                optimal_size = mid_size
                break
            elif size_ratio < target_ratio:
                min_size = mid_size + 1
                optimal_size = mid_size
            else:
                max_size = mid_size - 1

        return optimal_size

    def generate_number(self, char):
        image_size = self.config['image_size']

        img = self.create_base_image()
        draw = ImageDraw.Draw(img)

        # Get optimal font size for this character
        optimal_font_size = self._get_font_size_for_char(char, image_size)
        self.font.size = optimal_font_size

        # Get character dimensions with optimal font
        bbox = draw.textbbox((0, 0), char, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate exact center position
        x = (image_size - text_width) // 2 - bbox[0]  # Adjust for any negative left bearing
        y = (image_size - text_height) // 2 - bbox[1]  # Adjust for any negative top bearing

        # Draw the character
        draw.text((x, y), char, fill='white', font=self.font)

        return np.array(img)

    def generate_dot_pattern(self, number_str):
        image_size = self.config['image_size']

        img = self.create_base_image()
        draw = ImageDraw.Draw(img)
        number = int(number_str)

        # Calculate optimal dot size and spacing
        dot_diameter = max(2, image_size // 10)
        dot_radius = dot_diameter // 2

        # Calculate grid layout
        max_dots_per_row = min(3, number)
        rows = (number + max_dots_per_row - 1) // max_dots_per_row
        cols = min(max_dots_per_row, number)

        # Calculate total grid size
        grid_width = (cols - 1) * (dot_diameter * 2)
        grid_height = (rows - 1) * (dot_diameter * 2)

        # Calculate starting position to center the grid
        start_x = (image_size - grid_width) // 2
        start_y = (image_size - grid_height) // 2

        # Draw dots
        dots_placed = 0
        for row in range(rows):
            for col in range(cols):
                if dots_placed < number:
                    x = start_x + col * (dot_diameter * 2)
                    y = start_y + row * (dot_diameter * 2)

                    # Draw dot with anti-aliasing
                    draw.ellipse(
                        [x - dot_radius, y - dot_radius,
                         x + dot_radius, y + dot_radius],
                        fill='white'
                    )
                    dots_placed += 1

        return np.array(img)

    def generate_misc(self, type: str):
        """Generates an empty or filled image"""
        img = self.create_base_image()

        if type == 'filled':
            return ~np.array(img)
        else:
            return np.array(img)

    def save_example_images(self):
        """Save images for each alternative representation."""
        example_dir = self.output_dir / 'examples'
        example_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save number
        for num_type, symbol, img in self._generate_core():
            pil_img = Image.fromarray(img)
            pil_img.save(example_dir / f"{num_type}_{symbol}.png", format='PNG')
            pil_img.close()

    def save_dataset_to_mnist(self):
        """Save to MNIST format."""
        X, y = self._generate_dataset(num_samples_per_class=1)
        self._save_mnist_images(X)
        self._save_mnist_labels(y)


    def _generate_dataset(self, num_samples_per_class):
        """
        Generate a complete dataset of alternative number representations.

        Args:
            num_samples_per_class: Number of samples to generate per class

        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        X = []
        y = []

        # Generate defined characters
        for num_type, symbol, img in self._generate_core():
            for _ in range(num_samples_per_class):
                X.append(img)
                y.append(self.config['mappings'][num_type].index(symbol))


        return np.array(X), np.array(y)

    def _generate_core(self):
        # Generate and save symbols
        for num_type, symbols in self.config['mappings'].items():
            match num_type:
                case 'dots':
                    img_generator = self.generate_dot_pattern
                case 'misc':
                    img_generator = self.generate_misc
                case _:
                    img_generator = self.generate_number

            for symbol in symbols:
                # Defined character representation
                yield num_type, symbol, img_generator(symbol)

    def _save_mnist_images(self, images):
        """Save MNIST image format to disk."""
        image_path = self.output_dir / image_filename

        # Nothing to do as file exists
        if image_path.exists():
            return

        image_size = self.config['image_size']

        with open(image_path, 'wb') as f:
            # Write header
            f.write(struct.pack('>IIII',        # specify header format (big endian+4uint)
                                2051,           # magic number
                                len(images),        # number of images
                                image_size,    # number of rows
                                image_size))   # number of columns
            f.write(images.tobytes())

    def _save_mnist_labels(self, labels):
        """Save MNIST label format to disk."""
        label_path = self.output_dir / label_filename

        # Nothing to do as file exists
        if label_path.exists():
            return

        with open(label_path, 'wb') as f:
            # Write header
            f.write(struct.pack('>II',  # specify header format (big endian+2uint)
                    2049,               # magic number
                    len(labels)))           # number of labels
            f.write(labels.tobytes())