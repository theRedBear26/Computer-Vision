import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the U-Net Architecture (Standard for defect segmentation)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        
        x = self.up1(x2)
        # Skip connection
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up(x)
        return torch.sigmoid(self.out(x))

# 2. Create a synthetic "Crack" image for testing
def create_synthetic_crack(size=256):
    img = np.zeros((size, size, 3), dtype=np.float32)
    img += np.random.normal(0.5, 0.1, (size, size, 3)) # Concrete noise
    
    # Draw a fake crack
    x, y = 50, 0
    while y < size - 1:
        img[y, x] = [0.1, 0.1, 0.1] # Dark crack
        x += np.random.choice([-1, 0, 1])
        x = np.clip(x, 0, size-1)
        y += 1
    return img

if __name__ == "__main__":
    print("Initializing U-Net for Infrastructure Inspection...")
    model = UNet()
    
    # Generate test data
    test_img = create_synthetic_crack()
    # Convert to PyTorch Tensor (Batch Size, Channels, Height, Width)
    tensor_img = torch.tensor(test_img).permute(2, 0, 1).unsqueeze(0) 
    
    # Run Forward Pass
    with torch.no_grad():
        prediction = model(tensor_img)
    
    print(f"Input shape: {tensor_img.shape}")
    print(f"Output Mask shape: {prediction.shape}")
    
    # 3. Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(test_img)
    axes[0].set_title("Original Concrete Image")
    axes[0].axis('off')
    
    # Convert output mask to numpy for plotting
    mask = prediction.squeeze().numpy()
    axes[1].imshow(mask, cmap='Reds')
    axes[1].set_title("U-Net Predicted Defect Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()