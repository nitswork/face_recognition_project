import cv2
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Load the face detection model (Haar Cascade)
face_cap = cv2.CascadeClassifier(
    "C:/Users/nitya/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Load stored embeddings from pickle file
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Simple similarity function (cosine similarity)
def find_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Initialize video capture (camera)
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cap.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame
        face_img = video_data[y:y + h, x:x + w]
        # Generate the embedding for the cropped face
        try:
            embedding_obj = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)[0]
            test_embedding = embedding_obj["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            continue

        # Compare the generated embedding with stored embeddings
        best_match = None
        highest_similarity = -1  # Start from lowest possible similarity

        for name, emb in embeddings:
            sim = find_similarity(test_embedding, emb)
            if sim > highest_similarity:
                highest_similarity = sim
                best_match = name

        # Display the name and similarity below the face box
        if best_match:
            cv2.putText(video_data, f"{best_match} ({highest_similarity:.4f})", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the live video feed with face recognition
    cv2.imshow("Video Feed - Face Recognition", video_data)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_cap.release()
cv2.destroyAllWindows()
