import os
import pandas as pd

def generate_category_map(images_dir: str = "EMS/Images", output_path: str = "data/metadata/stimulus_categories.csv"):
    """
    Scans the subfolders of EMS/Images and generates a mapping from image name to category.
    """
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Map subfolder name to standard category name
    folder_to_cat = {
        "Manipulated Images": "Manipulated",
        "Natural Scenes": "Natural",
        "Social Scenes": "Social",
        "Synthetic Images": "Synthetic"
    }
    
    records = []
    for folder_name, cat_name in folder_to_cat.items():
        folder_path = os.path.join(images_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist.")
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({
                    "Image_Name": file_name,
                    "Category": cat_name
                })
                
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated category mapping for {len(df)} images at {output_path}")

if __name__ == "__main__":
    generate_category_map()
