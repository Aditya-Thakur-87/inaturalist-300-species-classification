import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import random
st.write("PyTorch version:", torch.__version__)
st.write("timm version:", timm.__version__)
# -----------------------------
# Model Definition (reuse your class)
# -----------------------------

class MultiModalEfficientNet(nn.Module):
    def __init__(self, num_classes, loc_feat_dim=64, backbone="tf_efficientnetv2_s"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.loc_mlp = nn.Sequential(
            nn.Linear(2, loc_feat_dim),
            nn.ReLU(),
            nn.Linear(loc_feat_dim, loc_feat_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(feat_dim + loc_feat_dim, num_classes)

    def forward(self, img, lat, lon):
        img_feat = self.backbone(img)
        loc_input = torch.stack([lat, lon], dim=1)
        loc_feat = self.loc_mlp(loc_input)

        fused = torch.cat([img_feat, loc_feat], dim=1)
        return self.classifier(fused)

    def unfreeze_last_block(self):
        last_block = self.backbone.blocks[-1]
        for param in last_block.parameters():
            param.requires_grad = True


# -----------------------------
# Load model & metadata
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache_resource
def load_model():
    model=MultiModalEfficientNet(num_classes=301,backbone="tf_efficientnetv2_s").to(device)
    checkpoint = torch.load("best_checkpoint_multimodal.pth",map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
@st.cache_data
def load_class_dir():
    return pd.read_csv("class_dir.csv")
@st.cache_data
def load_class_name():
    return pd.read_csv("class_name.csv")

model=load_model()
class_names=load_class_name()
class_dir=load_class_dir()


# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🐾 Wildlife Image Classifier (Multimodal)")
st.write("Upload an image and provide latitude & longitude to get predictions")


uploaded_file = st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])
# lat = st.number_input("Latitude",value=0.0,format="%.6f")
# lon =st.number_input("Longitude",value=0.0,format="%.6f")
lat_str = st.text_input("Latitude", value="0.0")
lon_str = st.text_input("Longitude", value="0.0")

# Convert to float safely
try:
    lat = float(lat_str)
    lon = float(lon_str)
except ValueError:
    st.error("Please enter valid numeric values for latitude and longitude.")
    st.stop()


if uploaded_file is not None:
    img =Image.open(uploaded_file).convert("RGB")
    st.image(img,caption="Uploaded Image",use_container_width=True)

    img_tensor = transform(img).unsqueeze(0).to(device)
    lat_tensor = torch.tensor([float(lat)],dtype=torch.float32).to(device)
    lon_tensor = torch.tensor([float(lon)],dtype=torch.float32).to(device)

    with torch.no_grad():
        output=model(img_tensor,lat_tensor,lon_tensor)
        probs = torch.softmax(output,dim=1)
        top5_probs,top5_class = probs.topk(5,1)

    st.subheader("Top 5 predictions:")

    predicted_indices = top5_class.cpu().numpy().tolist()[0]

    for i,idx in enumerate(predicted_indices):
        prob =top5_probs[0][i].item()
        name,common_name=class_names.loc[idx,["name","common_name"]].values
        st.write(f"{i+1}.**{name}** (**{common_name}**)- Probability: {prob:.2f}")

    dataset_path = r"C:\Users\thaku\jupyter_notebook _datasets\Wildlife_dataset\train_mini"
    st.subheader("Sample Images from Predicted Classes:")
    cols=st.columns(5)
    for i,idx in enumerate(predicted_indices):
        name, common_name = class_names.loc[idx, ["name", "common_name"]].values
        dir_name = class_dir.loc[idx,"image_dir_name"]
        full_dir = os.path.join(dataset_path,dir_name)
        if os.path.exists(full_dir):
            imgs =os.listdir(full_dir)
            if imgs:
                sample_img_path = os.path.join(full_dir,random.choice(imgs))
                sample_img = Image.open(sample_img_path).convert("RGB")
                with cols[i]:
                    st.image(sample_img, use_container_width=True)
                    st.markdown(f"**Species Name:** {name} ({common_name})")
st.write("Device:", device)