import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import time
import sys
import os # Import os for path checking

# === CONFIGURATION (IMPORTANT: MATCHES TRAINING CONFIG) ===
# These parameters *must* be identical to what was used during model training.
TARGET_SR = 48000  # RAVDESS sample rate
N_MFCC = 13        # Number of MFCCs
MAX_LEN = 173      # Fixed time steps for padding/trimming

# Audio recording parameters for live input
DURATION = 3       # seconds of audio to record at a time for each prediction
CHANNELS = 1       # Mono audio

# === DEFINE YOUR MODELS TO TEST ===
# Add all your model setups here. Each dictionary needs:
# 'name': A descriptive name for the model.
# 'model_path': Path to the .h5 model file.
# 'encoder_path': Path to the LabelEncoder .pkl file.
# 'scaler_path': Path to the StandardScaler .pkl file.
MODELS_TO_TEST = [
    {
        'name': 'Hybrid Features Model (MFCC+Delta+Delta2)',
        'model_path': 'emotion_cnn_model_hybrid_features.h5',
        'encoder_path': 'label_encoder.pkl',
        'scaler_path': 'scaler.pkl',
    },
    # Add other models here if you have them, e.g.:
    # {
    #     'name': 'MFCC + Delta Model (Your Original)',
    #     'model_path': 'emotion_cnn_model_mfcc_delta_only.h5', # Path to this model
    #     'encoder_path': 'label_encoder_mfcc_delta.pkl', # Path to its encoder
    #     'scaler_path': 'scaler_mfcc_delta.pkl', # Path to its scaler
    # },
    # {
    #     'name': 'Another Model Setup',
    #     'model_path': 'another_model.h5',
    #     'encoder_path': 'another_encoder.pkl',
    #     'scaler_path': 'another_scaler.pkl',
    # }
]

# === Feature Extraction Function (Copied directly) ===
def pad_or_trim(array, target_len):
    if array.shape[1] > target_len:
        return array[:, :target_len]
    else:
        pad_width = target_len - array.shape[1]
        return np.pad(array, ((0, 0), (0, pad_width)), mode='constant')

def extract_features_with_preprocessing(y, sr, n_mfcc, max_len):
    alpha = 0.97
    y_preemphasized = librosa.effects.preemphasis(y, coef=alpha)
    y_trimmed, _ = librosa.effects.trim(y_preemphasized, top_db=60)

    if len(y_trimmed) == 0:
        # This needs to match the exact feature dimension of the model being tested.
        # For the 'Hybrid Features Model' (MFCC+Delta+Delta2), this is n_mfcc * 3.
        # If testing other models with different feature sets, this might need adjustment
        # or the function needs to be split. For simplicity, assuming most models use this format.
        return np.zeros((max_len, n_mfcc * 3)) 

    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2) # This is the delta-delta part

    mfcc_padded = pad_or_trim(mfcc, max_len)
    delta_mfcc_padded = pad_or_trim(delta_mfcc, max_len)
    delta2_mfcc_padded = pad_or_trim(delta2_mfcc, max_len)

    combined_features = np.concatenate((mfcc_padded, delta_mfcc_padded, delta2_mfcc_padded), axis=0)

    return combined_features.T

# === Live Prediction Core Logic ===
def predict_live_emotion(audio_data, sr_live, model, encoder, scaler, n_mfcc, max_len):
    """Processes live audio data and predicts emotion using the provided model."""
    if sr_live != TARGET_SR:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr_live, target_sr=TARGET_SR)
        sr_live = TARGET_SR

    features = extract_features_with_preprocessing(audio_data, sr_live, n_mfcc, max_len)

    num_timesteps_single = features.shape[0]
    num_features_single = features.shape[1]
    
    # Check if the feature dimension matches the model's expected input dimension
    # This is crucial if you are testing models trained on different feature sets (e.g., MFCC+Delta vs MFCC+Delta+Delta2)
    # The `n_mfcc * 3` is for the current `extract_features_with_preprocessing`.
    # If a loaded model expects `n_mfcc * 2`, this `features` shape won't match.
    # A more robust solution would be to pass the exact feature dimension `n_features_expected` to this function,
    # or have a separate `extract_features` function per model type.
    # For simplicity, this current `extract_features_with_preprocessing` is designed for MFCC+Delta+Delta2.
    
    features_reshaped = features.reshape(-1, num_features_single)
    features_scaled = scaler.transform(features_reshaped)
    features_scaled = features_scaled.reshape(1, num_timesteps_single, num_features_single)

    prediction_probabilities = model.predict(features_scaled, verbose=0)[0]
    predicted_emotion_index = np.argmax(prediction_probabilities)
    predicted_emotion_label = encoder.inverse_transform([predicted_emotion_index])[0]

    return predicted_emotion_label, prediction_probabilities

# === Main Test Loop ===
def run_live_tests():
    for i, model_setup in enumerate(MODELS_TO_TEST):
        model_name = model_setup['name']
        model_path = model_setup['model_path']
        encoder_path = model_setup['encoder_path']
        scaler_path = model_setup['scaler_path']

        print(f"\n--- Testing Model {i+1}/{len(MODELS_TO_TEST)}: {model_name} ---")

        # Check if files exist
        if not all(os.path.exists(p) for p in [model_path, encoder_path, scaler_path]):
            print(f"❌ Error: Required files for '{model_name}' are missing.")
            print(f"  Model: {model_path} (Exists: {os.path.exists(model_path)})")
            print(f"  Encoder: {encoder_path} (Exists: {os.path.exists(encoder_path)})")
            print(f"  Scaler: {scaler_path} (Exists: {os.path.exists(scaler_path)})")
            cont = input("Press Enter to skip this model, or 'q' to quit: ")
            if cont.lower() == 'q':
                break
            continue

        # Load assets for the current model
        try:
            current_model = load_model(model_path)
            current_encoder = joblib.load(encoder_path)
            current_scaler = joblib.load(scaler_path)
            print("✅ Assets loaded for current model.")
        except Exception as e:
            print(f"❌ Failed to load assets for '{model_name}': {e}")
            cont = input("Press Enter to skip this model, or 'q' to quit: ")
            if cont.lower() == 'q':
                break
            continue

        print(f"\n🎙️ Speak for {DURATION} seconds to test '{model_name}'.")
        print("Type 'stop' and press Enter to stop testing this model.")
        print("Type 'next' and press Enter to move to the next model.")
        print("Type 'quit' and press Enter to exit the program.")

        while True:
            try:
                print("\n--- Recording... (Press Enter to process current recording) ---")
                # Use a non-blocking recording approach for better control
                recorded_chunks = []
                with sd.InputStream(samplerate=TARGET_SR, channels=CHANNELS, dtype='float32') as stream:
                    start_time = time.time()
                    while (time.time() - start_time) < DURATION:
                        chunk, overflowed = stream.read(int(TARGET_SR * 0.1)) # Read in small chunks
                        recorded_chunks.append(chunk)
                        time.sleep(0.01) # Small sleep to not busy-wait
                    
                    audio_data = np.concatenate(recorded_chunks, axis=0) # Combine chunks
                
                print("Processing...")

                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()

                predicted_label, probabilities = predict_live_emotion(
                    audio_data, TARGET_SR, current_model, current_encoder, current_scaler, N_MFCC, MAX_LEN
                )

                print(f"Predicted Emotion: {predicted_label.upper()}")
                print("Probabilities:")
                for idx in np.argsort(probabilities)[::-1]: # Sort by probability descending
                    label = current_encoder.inverse_transform([idx])[0]
                    prob = probabilities[idx]
                    print(f"  {label.capitalize()}: {prob:.2f}")

                user_input = input("\nAction ('test' again, 'next' model, 'quit'): ").lower().strip()
                if user_input == 'next':
                    break # Break from inner loop to go to next model
                elif user_input == 'quit':
                    sys.exit(0) # Exit program
                elif user_input == 'test':
                    continue # Test current model again
                else:
                    print("Invalid input. Continuing with current model test.")

            except KeyboardInterrupt:
                print("\nStopping current model test.")
                break # Exit current model's test loop
            except Exception as e:
                print(f"An error occurred during live prediction: {e}")
                traceback.print_exc() # Print full traceback for debugging
                cont_or_exit = input("Press Enter to try again, or 'q' to quit: ")
                if cont_or_exit.lower() == 'q':
                    sys.exit(0)
                continue # Try again with current model

    print("\nAll models tested. Exiting.")

if __name__ == "__main__":
    import traceback # Import here for easier use in except blocks
    run_live_tests()