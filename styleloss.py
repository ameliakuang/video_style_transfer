import zipfile
import os
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

zip_path = "frames_853913.zip"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Unzipped '{zip_path}' into '{output_dir}/'")

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

class VGGFeatures(nn.Module):
    def __init__(self, layers):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
        self.selected = nn.ModuleList([vgg[i] for i in range(max(layers) + 1)])
        self.eval()

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.selected):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features
    
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))  # Batch-wise matrix multiply
    return G / (h * w)

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])





style_image_path = "van_gogh.png"
style_img = Image.open(style_image_path).convert("RGB")
style_tensor = normalize_batch(image_transform(style_img).unsqueeze(0))
feature_layers = [1, 6, 11, 20, 29]
vgg = VGGFeatures(feature_layers)
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False
with torch.no_grad():
    style_features = vgg(style_tensor)
    style_grams = [gram_matrix(f) for f in style_features]
frame_folder = "frames_1"
loss_fn = nn.MSELoss()
with torch.no_grad():
    print("Style image self-loss (should be 0): {:.8f}\n".format(
        sum(loss_fn(gram_matrix(f), g).item() for f, g in zip(style_features, style_grams))
    ))
total_loss = 0.0
num_frames = 0

for filename in sorted(os.listdir(frame_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame_path = os.path.join(frame_folder, filename)
        frame_img = Image.open(frame_path).convert("RGB")
        frame_tensor = normalize_batch(image_transform(frame_img).unsqueeze(0))

        with torch.no_grad():
            frame_features = vgg(frame_tensor)

            style_loss = 0.0
            print(f"Frame: {filename} + {os.path.basename(style_image_path)}")
            for i, (f_f, g_s) in enumerate(zip(frame_features, style_grams)):
                g_f = gram_matrix(f_f)
                layer_loss = loss_fn(g_f, g_s).item()
                print(f"  Layer {i} style loss: {layer_loss:.6f}")
                style_loss += layer_loss

            print(f"Style loss for frame: {style_loss:.6f}\n")
            total_loss += style_loss
            num_frames += 1

avg_loss = total_loss / num_frames if num_frames > 0 else 0.0
print(f"Average Style Consistency Loss over {num_frames} frames: {avg_loss:.6f}")




