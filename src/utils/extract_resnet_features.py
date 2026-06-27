import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

def main():
    print("=" * 70)
    print(" VISUAL FEATURE EXTRACTION (ResNet50 Baseline)")
    print("=" * 70)

    # 1. Paths
    parquet_path = "data/processed/clean_fixations.parquet"
    images_dir = "EMS/Images"
    output_path = "data/external/feature_dict_ResNet50.npy"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Clean fixations parquet not found at {parquet_path}. "
            "Please run Tier 1 preprocessing first:\n"
            "  python src/tier1_preprocessing/preprocess.py"
        )

    # 2. Build image lookup
    image_paths = {}
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths[f] = os.path.join(root, f)

    print(f"Found {len(image_paths)} stimulus images in {images_dir}.")
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in {images_dir}. Check EMS path.")

    # 3. Load ResNet50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pre-trained ResNet50 on device: {device}...")
    
    # Load ResNet50 without the final classification layers to get spatial features
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Extract only the backbone layers (features output is [2048, H/32, W/32])
    backbone = nn.Sequential(*list(resnet.children())[:-2]).to(device)
    backbone.eval()

    # Image transform (ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. Extract feature maps for all unique images
    print("Extracting feature maps for all stimulus images...")
    df = pd.read_parquet(parquet_path)
    unique_images = df['IMAGE'].unique()
    
    feature_maps = {}
    with torch.no_grad():
        for img_name in tqdm(unique_images, desc="Extracting image features"):
            path = image_paths.get(img_name)
            if path is None:
                print(f"\nWarning: Image {img_name} not found in {images_dir}. Using zero features.")
                feature_maps[img_name] = torch.zeros((2048, 24, 32), device=device)
                continue
            
            try:
                img = Image.open(path).convert('RGB')
                # Resize to standard EMS resolution (1024x768)
                img = img.resize((1024, 768))
                img_t = transform(img).unsqueeze(0).to(device) # [1, 3, 768, 1024]
                
                # Forward pass through backbone
                f_map = backbone(img_t) # [1, 2048, 24, 32] (downsampled by 32)
                feature_maps[img_name] = f_map.squeeze(0) # [2048, 24, 32]
            except Exception as e:
                print(f"\nError processing image {img_name} at {path}: {e}")
                feature_maps[img_name] = torch.zeros((2048, 24, 32), device=device)

    # 5. Map features to fixations per trial
    print("Mapping visual features to subject fixations...")
    grouped = df.groupby(['Subject_ID', 'IMAGE'])
    
    feature_dict = {}
    for (sub_id, img_name), df_trial in tqdm(grouped, desc="Mapping to trials"):
        # Sort chronologically to match graph_builder.py
        df_sorted = df_trial.sort_values('FIX_INDEX').reset_index(drop=True)
        
        f_map = feature_maps[img_name].cpu().numpy() # [2048, 24, 32]
        
        trial_feats = []
        for _, row in df_sorted.iterrows():
            fx = row['FIX_X']
            fy = row['FIX_Y']
            
            # Map screen coordinates (1024x768) to feature map coordinates (32x24)
            # Factor is 32 (1024/32 = 32, 768/24 = 32)
            ix = int(np.clip(round(fx / 32.0), 0, 31))
            iy = int(np.clip(round(fy / 32.0), 0, 23))
            
            feat_vec = f_map[:, iy, ix] # [2048]
            trial_feats.append(feat_vec)
            
        feature_dict[(sub_id, img_name)] = np.array(trial_feats, dtype=np.float32)

    # 6. Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, feature_dict)
    print(f"\nSuccessfully extracted ResNet50 features. Saved to {output_path}")
    print(f"Total trials mapped: {len(feature_dict)}")
    print(f"Feature vector dimension: 2048")
    print("=" * 70)

if __name__ == "__main__":
    main()
