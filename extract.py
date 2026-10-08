import os
import shutil

def reduce_to_one_image(source_dir, target_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')

    if not os.path.exists(source_dir):
        print(f"[Error] Source directory '{source_dir}' does not exist.")
        return

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get all subdirectories (e.g., person folders)
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    print(f"Found {len(subdirs)} folders. Extracting 1 image per folder...")

    for subdir_name in subdirs:
        source_person_path = os.path.join(source_dir, subdir_name)
        target_person_path = os.path.join(target_dir, subdir_name)
        
        # Sort files to ensure we grab the first one consistently
        files = sorted(os.listdir(source_person_path))
        
        for file in files:
            if file.lower().endswith(valid_exts):
                # We found the first valid image
                source_file_path = os.path.join(source_person_path, file)
                target_file_path = os.path.join(target_person_path, file)
                
                # Create the subdirectory in the new target folder
                os.makedirs(target_person_path, exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_file_path, target_file_path)
                print(f"[Copied] {subdir_name}/{file}")
                
                break  # Stop after copying exactly 1 image

    print("\nExtraction complete! Check the new directory.")

if __name__ == "__main__":
    # Change these paths to match your system
    SOURCE_DIRECTORY = r"C:\Users\lenovo\Desktop\dataset\Original Images"
    TARGET_DIRECTORY = r"C:\Users\lenovo\Desktop\dataset\Reduced Images"
    
    reduce_to_one_image(SOURCE_DIRECTORY, TARGET_DIRECTORY)