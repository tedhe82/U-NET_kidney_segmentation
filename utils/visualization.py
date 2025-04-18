import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_sample(image, mask, prediction=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.axis('off')
    
    if prediction is not None:
        plt.subplot(133)
        plt.imshow(prediction.squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results(epoch, train_losses, val_losses, save_path='results.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close() 