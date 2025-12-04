import cv2
import os
from deepface import DeepFace
from fer import FER
from collections import Counter
import warnings

def predict_photos_emotion(folder_path="photos", limit=5):
    warnings.filterwarnings("ignore")

    emo_detector = FER(mtcnn=True)
    results = []

    # Load up to 5 valid image files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    files = files[:limit]  # Only first 5 files

    for file in files:
        image_path = os.path.join(folder_path, file)
        img = cv2.imread(image_path)

        if img is None:
            results.append({"file": file, "emotion": "Invalid image", "image": None})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            results.append({"file": file, "emotion": "No face detected", "image": img})
            continue

        # Use only the first detected face
        x, y, w, h = faces[0]
        face_roi = img[y:y + h, x:x + w]

        # Skip empty ROI
        if face_roi.size == 0:
            results.append({"file": file, "emotion": "Face ROI invalid", "image": img})
            continue

        predictions = []

        # DeepFace Prediction
        try:
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            predictions.append(analysis["dominant_emotion"])
        except:
            pass

        # FER Prediction
        try:
            fer_result = emo_detector.detect_emotions(face_roi)
            if fer_result:
                dominant_fer = max(fer_result[0]['emotions'], key=fer_result[0]['emotions'].get)
                predictions.append(dominant_fer)
        except:
            pass

        # Final decision
        if predictions:
            final_emotion = Counter(predictions).most_common(1)[0][0]
        else:
            final_emotion = "Could not detect"

        results.append({
            "file": file,
            "emotion": final_emotion,
            "image": img
        })

    return results

results = predict_photos_emotion()

for res in results:
    print("File:", res["file"])
    print("Emotion:", res["emotion"])

    # Show the image
    if res["image"] is not None:
        cv2.imshow(res["file"], res["image"])
        cv2.waitKey(0)
