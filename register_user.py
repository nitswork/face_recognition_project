import cv2
from deepface import DeepFace
from db_utils import get_connection
import time

# Load Haar cascade for face detection
face_cap = cv2.CascadeClassifier(
    "C:/Users/nitya/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Ask user for their name
user_name = input("Enter your name: ")

# Start webcam
video_cap = cv2.VideoCapture(0)

print("Capturing 5 face samples. Please face the camera...")

captured = 0
max_samples = 5
face_embeddings = []

while captured < max_samples:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle for feedback
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face
        face_img = frame[y:y + h, x:x + w]

        try:
            # Get embedding
            embedding_obj = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)[0]
            embedding = embedding_obj["embedding"]
            face_embeddings.append(embedding)
            captured += 1
            print(f"Captured sample {captured}/{max_samples}")
            time.sleep(1)  # Slight pause to allow repositioning
        except Exception as e:
            print(f"Error generating embedding: {e}")
            continue

    cv2.imshow("Registering Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Registration interrupted.")
        break

# Save to DB
if face_embeddings:
    conn = get_connection()
    cur = conn.cursor()
    for emb in face_embeddings:
        cur.execute("INSERT INTO face_data (filename, embedding) VALUES (%s, %s)", (user_name, emb))
    conn.commit()
    cur.close()
    conn.close()
    print(f"\nRegistered {captured} face embeddings for '{user_name}'.")

# Cleanup
video_cap.release()
cv2.destroyAllWindows()
