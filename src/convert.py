import os
from pydub import AudioSegment

INPUT_PATH = "data/organized/"
OUTPUT_PATH = "data/wav/"

def convert_ogg_to_wav():
    """Converts all .ogg files to .wav format."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for species in os.listdir(INPUT_PATH):
        species_folder = os.path.join(INPUT_PATH, species)
        output_species_folder = os.path.join(OUTPUT_PATH, species)
        os.makedirs(output_species_folder, exist_ok=True)

        for file in os.listdir(species_folder):
            if file.endswith(".ogg"):
                ogg_path = os.path.join(species_folder, file)
                wav_filename = file.replace(".ogg", ".wav")
                wav_path = os.path.join(output_species_folder, wav_filename)

                # Convert to WAV
                audio = AudioSegment.from_ogg(ogg_path)
                audio.export(wav_path, format="wav")

                print(f"✅ Converted: {ogg_path} → {wav_path}")

if __name__ == "__main__":
    convert_ogg_to_wav()
 