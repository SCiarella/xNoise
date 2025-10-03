import os
import sys
import random
import csv
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image, make_grid
from textwrap import wrap
import gc
import time
import clip

from ddpm_clip.models import UNet, DDPM, sample_w, visualize_diffusion_process
from ddpm_clip.utils import save_animation, generation_image, to_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == 'cuda', "CPU not supported, please run on a machine with CUDA"

# Set new HOME directory to scratch space for Das6
new_home = '/var/scratch/ciarella/xAI'
os.environ['HOME'] = new_home
os.makedirs(new_home, exist_ok=True)
print(f"Changed HOME directory to: {os.environ['HOME']}")

# Clean GPU
gc.collect()
torch.cuda.empty_cache()

# Load CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.eval()
CLIP_FEATURES = 512

# Dataset configuration
dataset_root = "../data/tiny-imagenet-200"
train_paths = glob.glob(f"{dataset_root}/train/*/images/*.JPEG")
val_paths = glob.glob(f"{dataset_root}/val/images/*.JPEG")
data_paths = train_paths

ndata = 100000
print(f"Found {len(data_paths)} total images, but using only {ndata}")
random.seed(11)
random.shuffle(data_paths)
data_paths = data_paths[:ndata]

# Process CLIP encodings
csv_path = 'clip.csv'
batch_size = 64

existing_paths = set()
if os.path.exists(csv_path):
    print(f"Found existing CSV file: {csv_path}")
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row:
                existing_paths.add(row[0])
    print(f"Found {len(existing_paths)} already processed images")
else:
    print(f"Creating new CSV file: {csv_path}")

remaining_paths = [path for path in data_paths if path not in existing_paths]
print(f"Processing {len(remaining_paths)} new images (skipping {len(existing_paths)} already done)")

if len(remaining_paths) > 0:
    all_rows = []
    start_time = time.time()

    for batch_start in range(0, len(remaining_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_paths))
        batch_paths = remaining_paths[batch_start:batch_end]
        
        batch_imgs = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                processed_img = clip_preprocess(img)
                batch_imgs.append(processed_img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if batch_imgs:
            clip_imgs_batch = torch.stack(batch_imgs).to(device)
            
            with torch.no_grad():
                labels_batch = clip_model.encode_image(clip_imgs_batch)
            
            for path, label in zip(valid_paths, labels_batch):
                all_rows.append([path] + label.cpu().tolist())
            
            if (batch_start // batch_size + 1) % 20 == 0:
                torch.cuda.empty_cache()
        
        if (batch_start // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = batch_end / elapsed
            remaining = (len(remaining_paths) - batch_end) / rate if rate > 0 else 0
            print(f"Processed {batch_end}/{len(remaining_paths)} new images... "
                  f"Rate: {rate:.1f} imgs/sec, ETA: {remaining:.0f}s")

    if all_rows:
        print(f"Appending {len(all_rows)} new entries to CSV...")
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(all_rows)
        
        total_time = time.time() - start_time
        print(f"Completed! Processed {len(all_rows)} new images in {total_time:.1f}s")

# Image transformations
IMG_CH = 3
IMG_SIZE = 32
BATCH_SIZE = 16
INPUT_SIZE = (IMG_CH, IMG_SIZE, IMG_SIZE)

pre_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

random_transforms = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
])

# Dataset class
class MyDataset(Dataset):
    def __init__(self, csv_path, preprocessed_clip=True):
        self.imgs = []
        self.preprocessed_clip = preprocessed_clip
        if preprocessed_clip:
            self.labels = torch.empty(
                len(data_paths), CLIP_FEATURES, dtype=torch.float, device=device
            )
        
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(reader):
                img = Image.open(row[0]).convert('RGB')
                self.imgs.append(pre_transforms(img).to(device))
                if preprocessed_clip:
                    label = [float(x) for x in row[1:]]
                    self.labels[idx, :] = torch.FloatTensor(label).to(device)

    def __getitem__(self, idx):
        img = random_transforms(self.imgs[idx])
        if self.preprocessed_clip:
            label = self.labels[idx]
        else:
            batch_img = img[None, :, :, :]
            encoded_imgs = clip_model.encode_image(clip_preprocess(batch_img))
            label = encoded_imgs.to(device).float()[0]
        return img, label

    def __len__(self):
        return len(self.imgs)

# Create dataset and dataloader
train_data = MyDataset(csv_path, preprocessed_clip=True)
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Initialize model
T = 400
B_start = 0.0001
B_end = 0.02
B = torch.linspace(B_start, B_end, T).to(device)

ddpm = DDPM(B, device)
model = UNet(
    T, IMG_CH, IMG_SIZE, down_chs=(256, 256, 512), t_embed_dim=8, c_embed_dim=CLIP_FEATURES
)
print("Num params: ", sum(p.numel() for p in model.parameters()))

# Enable TF32 for faster training on Ampere GPUs and newer
torch.set_float32_matmul_precision('high')

model_tinyimg = torch.compile(model.to(device))

# Sampling function
def sample_tinyimg(text_list, w_values=None):
    text_tokens = clip.tokenize(text_list).to(device)
    c = clip_model.encode_text(text_tokens).float()
    x_gen, x_gen_store = sample_w(model, ddpm, INPUT_SIZE, T, c, device, w_tests=w_values)
    return x_gen, x_gen_store

# Training parameters
epochs = 100
c_drop_prob = 0.1
lrate = 1e-4
save_dir = "../images/"
checkpoint_path = "../model_checkpoint"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
    if checkpoint_files:
        latest_epoch = max([int(f.split('.')[0]) for f in checkpoint_files])
        latest_checkpoint = os.path.join(checkpoint_path, f"{latest_epoch}.pth")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
else:
    print("Initializing model")

if start_epoch >= epochs:
    print(f"Already trained to epoch {epochs}")
else:
    model.train()
    for epoch in range(start_epoch, epochs):
        gc.collect()
        torch.cuda.empty_cache()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).float()
            x, c = batch
            c_mask = ddpm.get_context_mask(c, c_drop_prob)
            loss = ddpm.get_loss(model_tinyimg, x, t, c, c_mask)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item()}")
        if epoch % 5 == 0 or epoch == int(epochs - 1):
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, f"{checkpoint_path}/{epoch}.pth")
            text_list = ["fish", "tree", "sunset"]
            w_list = [-2, -1, 0, 1, 2]
            x_gen, x_gen_store = sample_tinyimg(text_list, w_list)
            generation_image(x_gen, text_list, w=w_list, save_path=save_dir+f"image_ep{epoch:02}.png")
            print(f"Saved images in {save_dir} for epoch {epoch}")

# Final generation
print("Generating final samples...")
model.eval()
text_list = ["A fish", "A tree", "A sunset"]
w_values = [-2, -1, 0, 1, 2]
x_gen, x_gen_store = sample_tinyimg(text_list, w_values)
generation_image(x_gen, text_list, w=w_values, save_path=f"{save_dir}final_generation.png")

# Generate animation
grids = [generation_image(x_gen.cpu(), text_list, w=w_values) for x_gen in x_gen_store]
save_animation(grids, "../images/test.gif")
print("Training and generation complete!")
