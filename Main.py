import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
from db_utils import get_connection

# Load the face detection model (Haar Cascade)
face_cap = cv2.CascadeClassifier(
    "C:/Users/nitya/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

def load_embeddings_from_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT filename, embedding FROM face_data")
    data = cur.fetchall()
    cur.close()
    conn.close()
    # Convert embeddings from list to proper format if needed
    # psycopg2 returns the array as a Python list already
    return [(filename, embedding) for filename, embedding in data]

# Load embeddings from DB
embeddings = load_embeddings_from_db()

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

    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_img = video_data[y:y + h, x:x + w]

        try:
            embedding_obj = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)[0]
            test_embedding = embedding_obj["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            continue

        best_match = None
        highest_similarity = -1

        for name, emb in embeddings:
            sim = find_similarity(test_embedding, emb)
            if sim > highest_similarity:
                highest_similarity = sim
                best_match = name

        # Threshold example (adjust 0.6 as needed)
        if best_match and highest_similarity > 0.6:
            display_text = f"{best_match} ({highest_similarity:.4f})"
        else:
            display_text = "Unknown"

        cv2.putText(video_data, display_text, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video Feed - Face Recognition", video_data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
