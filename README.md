# U-NET Kidney Segmentation

This project implements a U-NET architecture for automatic kidney segmentation from MRI images. The implementation is in PyTorch and provides end-to-end training and prediction pipelines.

## Project Structure

U-NET_kidney_segmentation/
├── data/
│ ├── raw/ # For storing raw MRI images
│ └── processed/ # For preprocessed data
├── models/
│ ├── init.py
│ └── unet.py # U-NET implementation
├── utils/
│ ├── init.py
│ ├── data_loader.py # Data loading and preprocessing
│ └── visualization.py # Visualization utilities
├── train.py # Training script
├── predict.py # Prediction script
└── requirements.txt # Project dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/U-NET_kidney_segmentation.git
cd U-NET_kidney_segmentation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place your MRI images in `data/raw/images/`
2. Place corresponding segmentation masks in `data/raw/masks/`
3. Ensure masks are binary images (255 for kidney, 0 for background)

### Training

To train the model:
```bash
python train.py
```

The model will be saved as `best_model.pth` whenever it achieves better validation loss.

### Prediction

To run prediction on new images:
```bash
python predict.py
```

## Model Architecture

The implemented U-NET architecture consists of:
- Encoder path with 5 levels of double convolution blocks
- Decoder path with skip connections
- Input size: 256x256 grayscale images
- Output: Binary segmentation mask

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL
- numpy
- matplotlib
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.