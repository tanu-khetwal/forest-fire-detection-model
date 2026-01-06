import os
import shutil
import random

print("Starting to create validation set...")

# --- Configuration ---
SOURCE_DIR = 'dataset/train'
VALID_DIR = 'dataset/validation'
SPLIT_RATIO = 0.2  # 20% of images will be moved for validation
CLASSES = ['fire', 'no_fire']
# ---------------------

random.seed(42) # Use a fixed seed for reproducible results

if not os.path.exists(SOURCE_DIR):
    print(f"Error: Source directory '{SOURCE_DIR}' not found.")
    print("Please make sure your 'fire' and 'no_fire' images are inside 'dataset/train/'")
    exit()

# Create the validation directory structure
os.makedirs(VALID_DIR, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(VALID_DIR, cls), exist_ok=True)
    print(f"Created directory: {os.path.join(VALID_DIR, cls)}")

# --- Main splitting logic ---
for cls in CLASSES:
    source_class_path = os.path.join(SOURCE_DIR, cls)
    validation_class_path = os.path.join(VALID_DIR, cls)
    
    # Check if the source class folder exists
    if not os.path.exists(source_class_path):
        print(f"Warning: Class folder '{source_class_path}' not found. Skipping.")
        continue
        
    # Get a list of all image files
    all_files = [f for f in os.listdir(source_class_path) 
                 if os.path.isfile(os.path.join(source_class_path, f))]
    
    # Shuffle the list randomly
    random.shuffle(all_files)
    
    # Calculate the number of files to move
    split_point = int(len(all_files) * SPLIT_RATIO)
    
    # Get the list of files to move
    validation_files = all_files[:split_point]
    
    print(f"\nProcessing class: '{cls}'")
    print(f"Total images: {len(all_files)}")
    print(f"Moving {len(validation_files)} images to validation...")
    
    # Move the files
    for file_name in validation_files:
        src_path = os.path.join(source_class_path, file_name)
        dst_path = os.path.join(validation_class_path, file_name)
        
        try:
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"Error moving file {src_path}: {e}")
            
    print(f"Finished moving files for class '{cls}'.")
    print(f"Remaining in train: {len(all_files) - len(validation_files)}")

print("\nValidation set creation complete!")
print(f"Your 'dataset/train' folder now contains 80% of the images.")
print(f"Your 'dataset/validation' folder now contains 20% of the images.")