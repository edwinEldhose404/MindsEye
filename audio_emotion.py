import keras
import librosa
import numpy as np
import sounddevice as sd
import time

# --- Configuration ---
MODEL_PATH = 'emotion_cnn_model_augmented_combined_features.h5'
EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]
# --- Audio Settings ---
SAMPLE_RATE = 44100  # Must match the rate used during training (22050 * 2)
DURATION = 3         # Record for 3 seconds

def extract_features_from_audio(audio_data, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512, max_pad_len=173):
    """
    Extracts features directly from an audio data array.
    This is modified to not use a file path.
    """
    try:
        # Extract base MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=num_mfcc)
        
        # Extract delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Stack them together
        features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

        # Pad or truncate to the fixed length
        if features.shape[1] > max_pad_len:
            features = features[:, :max_pad_len]
        else:
            pad_width = max_pad_len - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return features.T
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def get_live_emotion(model, labels):
    """
    Records audio from the microphone, processes it, and returns the predicted emotion.
    
    Returns:
        str: The name of the predicted emotion.
    """
    # 1. Countdown and record audio
    print("Get ready to speak...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording for 3 seconds...")
    # The recording is a numpy array
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # 2. Process the audio
    print("Processing audio...")
    # If recording is stereo, convert to mono by averaging channels
    if recording.ndim > 1 and recording.shape[1] > 1:
        audio_data = np.mean(recording, axis=1)
    else:
        audio_data = recording.flatten()
        
    # 3. Extract features
    features = extract_features_from_audio(audio_data, SAMPLE_RATE)
    
    if features is None:
        return "Feature extraction failed."

    # 4. Reshape for model prediction and predict
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    
    # 5. Return the predicted emotion label
    predicted_emotion = labels[predicted_index]
    return predicted_emotion


# --- Main execution block ---
if __name__ == '__main__':
    try:
        # Load the pre-trained model
        emotion_model = keras.models.load_model(MODEL_PATH)

        # Get the prediction from the live audio
        final_emotion = get_live_emotion(emotion_model, EMOTION_LABELS)
        
        # Print the final result
        print("\n" + "="*30)
        print(f"✨ Predicted Emotion: {final_emotion.upper()} ✨")
        print("="*30)

    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")