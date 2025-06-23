import pickle
from deepface import DeepFace
print("DeepFace is installed and imported successfully.")

import cv2
import os

folder = 'stored-faces'
embeddings = []

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    try:
        # Use DeepFace to get embedding with enforce_detection=False
        embedding_obj = DeepFace.represent(img_path=path, model_name='Facenet', enforce_detection=False)[0]
        embeddings.append((file, embedding_obj["embedding"]))
    except Exception as e:
        print(f"Error with {file}: {e}")

# Save embeddings to file
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
