import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2 
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.upconv4(bn)
        up4 = torch.cat([up4, d4], dim=1)
        up4 = self.up4(up4)

        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, d3], dim=1)
        up3 = self.up3(up3)

        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.up2(up2)

        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, d1], dim=1)
        up1 = self.up1(up1)

        out = self.final_conv(up1)

        return out

# --- Load model ---

@st.cache_resource
def load_model():
    model = UNet()  # Make sure UNet() matches the model you trained
    model.load_state_dict(torch.load("unet_segmentation.pth", map_location=torch.device('cpu') ))
    #model = torch.load("unet_segmentation.pth", map_location=torch.device('cpu') )
    model.eval()
    return model

# --- Preprocess image ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),             # Converts to [C x H x W] and normalizes to [0,1]
    ])
    return transform(image).unsqueeze(0)   # Add batch dimension

# --- Postprocess output mask ---
def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8)  # Thresholding
    return mask

# --- Streamlit interface ---
st.title("🧠 Brain Tumor Segmentation")
st.write("Upload a brain MRI image to generate the predicted segmentation mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Processing...'):
        model = load_model()
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
        mask = postprocess_mask(output)
    #binary_mask = (mask > 0.5).astype(np.uint8) * 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_np=np.array(image)
    # Copy the original image so we don't modify the original
    outlined_image = image_np.copy()

    # Draw contours on the image in red
    cv2.drawContours(outlined_image, contours, -1, color=(255, 0, 0), thickness=2)
    # Display side-by-side
    col1, col2 = st.columns(2)
    col1.image(image.resize((256, 256)), caption="Original Image", use_column_width=True)
    col2.image(outlined_image , caption="Segmented image", use_column_width=True, clamp=True)
