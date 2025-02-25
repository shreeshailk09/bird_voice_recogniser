import os
import pandas as pd
import shutil

# Paths
CSV_PATH = "data/raw/train_metadata.csv"  # Metadata CSV
AUDIO_PATH = "data/raw/train_audio/" 
OUTPUT_PATH = "data/organized/"  

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

def organize_audio_files():
    """Sorts BirdCLEF audio files into species folders."""
    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        species = row["primary_label"]  
        filename = row["filename"]  

        # Correct file paths
        src = os.path.join(AUDIO_PATH, filename)  # Source file
        species_folder = os.path.join(OUTPUT_PATH, species)  # Destination folder
        dst = os.path.join(species_folder, os.path.basename(filename))  # Correct filename

        # Create the species folder if it doesn't exist
        os.makedirs(species_folder, exist_ok=True)

        # Check if the source file exists before moving
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved: {src} → {dst}")
        else:
            print(f"⚠️ File not found: {src}")

    print("✅ All files organized successfully!")

if __name__ == "__main__":
    organize_audio_files()
