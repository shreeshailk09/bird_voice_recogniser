import torch
import numpy as np
import librosa
import json
import sys
import os
from model_training import BirdClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "models/bird_classifier_best.pth"
FEATURES_FILE = "data/processed/features.npy"

data = np.load(FEATURES_FILE, allow_pickle=True).item()
label_map = data["label_map"]
feature_size = 141  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map)
model = BirdClassifier(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def extract_features(audio_path, max_length=5):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=max_length)
        if len(y) < sr * max_length:
            y = np.pad(y, (0, sr * max_length - len(y)))
        else:
            y = y[:sr * max_length]

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_size)
        mfccs = np.mean(mfccs.T, axis=0)  

        if mfccs.shape[0] < feature_size:
            mfccs = np.pad(mfccs, (0, feature_size - mfccs.shape[0]))  
        elif mfccs.shape[0] > feature_size:
            mfccs = mfccs[:feature_size]

        return mfccs.reshape(1, -1)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def predict_bird(audio_path):
    features = extract_features(audio_path)
    if features is None:
        return "Failed to process audio file."
    with torch.no_grad():
        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        output = model(input_tensor)
        predicted_label = torch.argmax(output).item()
    return label_map.get(predicted_label, f"Unknown label: {predicted_label}")

if __name__ == "__main__":
    test_audio = "data/raw/sample.wav"
    predicted_species = predict_bird(test_audio)
    print(f"Predicted Bird Species: {predicted_species}")

