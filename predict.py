import torch
from models.unet import UNet
from PIL import Image
import numpy as np
from torchvision import transforms
from utils.visualization import plot_sample
import argparse

def predict_mask(model, image_path, device):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert prediction to numpy array
    prediction = prediction.cpu().squeeze().numpy()
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    
    return prediction, np.array(image)

def main():
    parser = argparse.ArgumentParser(description='Predict kidney segmentation mask')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to the trained model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    # Make prediction
    prediction, original_image = predict_mask(model, args.image_path, device)
    
    # Display results
    plot_sample(original_image, prediction)

if __name__ == '__main__':
    main() 