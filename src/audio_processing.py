import os
import librosa
import numpy as np
import random
import pandas as pd
import json
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

CSV_PATH = "data/raw/train_metadata.csv"
DATASET_PATH = "data/wav/"
OUTPUT_FILE = "data/processed/features.npy"

df = pd.read_csv(CSV_PATH)
species_mapping = dict(zip(df["primary_label"], df["common_name"]))

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
    Gain(min_gain_db=-5, max_gain_db=5, p=0.3),
])

def augment_audio(y, sr):
    if random.random() < 0.5:
        start = random.randint(0, max(1, len(y) - sr * 2))
        y[start: start + sr * 2] = augment(samples=y[start: start + sr * 2], sample_rate=sr)
    return y

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        max_length = min(10, duration)

        if len(y) < sr * max_length:
            y = np.pad(y, (0, int(sr * max_length - len(y))))
        else:
            y = y[:int(sr * max_length)]

        y = augment_audio(y, sr)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        features = np.hstack([
            np.mean(mfccs, axis=1), 
            np.mean(delta_mfccs, axis=1), 
            np.mean(delta2_mfccs, axis=1), 
            np.mean(spectral_centroid, axis=1), 
            np.mean(chroma, axis=1), 
            np.mean(spectral_contrast, axis=1),
            np.mean(rolloff, axis=1)
        ])

        print(f"✅ Feature vector size: {features.shape}")  # 🔹 Debug feature size

        return features.tolist()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def balance_dataset(features, labels):
    ros = RandomOverSampler(sampling_strategy="auto")
    features_resampled, labels_resampled = ros.fit_resample(np.array(features), np.array(labels))
    return features_resampled.tolist(), labels_resampled.tolist()

def create_dataset():
    dataset = {"features": [], "labels": [], "label_map": {}}
    label_id = 0

    for species in os.listdir(DATASET_PATH):
        species_path = os.path.join(DATASET_PATH, species)
        if os.path.isdir(species_path):
            common_name = species_mapping.get(species, species)
            dataset["label_map"][label_id] = common_name

            for file in os.listdir(species_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(species_path, file)
                    features = extract_features(file_path)
                    if features:
                        dataset["features"].append(features)
                        dataset["labels"].append(label_id)

            label_id += 1

    if not dataset["features"]:
        print("Error: No features extracted! Check if data exists in the correct folder.")
        return

    dataset["features"], dataset["labels"] = balance_dataset(dataset["features"], dataset["labels"])

    np.save(OUTPUT_FILE, dataset)
    print(f"Dataset saved at {OUTPUT_FILE} with {len(dataset['features'])} samples.")

if __name__ == "__main__":
    create_dataset()
