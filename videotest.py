import cv2
from deepface import DeepFace
from fer import FER 
import time
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# Model 1: DeepFace (uses its default emotion model)
# Model 2: FER (Face Emotion Recognition)
emo_detector = FER(mtcnn=True)

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

ANALYSIS_DURATION = 5 
start_time = time.time()
all_detected_emotions = [] 

print(f"Starting ensemble emotion detection for {ANALYSIS_DURATION} seconds...")

while (time.time() - start_time) < ANALYSIS_DURATION:
    ret, frame = cap.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (25_5, 0, 0), thickness=3)
        face_roi = frame[y:y + h, x:x + w]
        
        current_face_predictions = []

        # --- Prediction from Model 1: DeepFace ---
        try:
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            current_face_predictions.append(analysis['dominant_emotion'])
        except Exception:
            pass 

        # --- Prediction from Model 2: FER ---
        try:
            result = emo_detector.detect_emotions(face_roi)
            if result:
                # Get the emotion with the highest score from FER's output
                dominant_emotion_fer = max(result[0]['emotions'], key=result[0]['emotions'].get)
                current_face_predictions.append(dominant_emotion_fer)
        except Exception:
            pass 
        
        if current_face_predictions:
            majority_emotion = Counter(current_face_predictions).most_common(1)[0][0]
            all_detected_emotions.append(majority_emotion)
        
            display_text = f"VOTE: {majority_emotion}"
            cv2.putText(frame, display_text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Ensemble Emotion Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n-------------------------------------------")
print(f"Analysis complete after {ANALYSIS_DURATION} seconds.")

if all_detected_emotions:
    overall_majority = Counter(all_detected_emotions).most_common(1)[0][0]
    
    print(f"\nOverall Majority Emotion: {overall_majority}")
    print(f"Frame-by-frame majority votes: {all_detected_emotions}")
else:
    print("\nNo emotions were detected in the time frame.")

print("-------------------------------------------")