# import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from db_utils import get_connection

# # Load stored embeddings
# with open("embeddings.pkl", "rb") as f:
#     embeddings = pickle.load(f)

# # Generate embedding for test image
# test_img_path = ""  # Replace with your image path
# test_embedding_obj = DeepFace.represent(img_path=test_img_path, model_name='Facenet', enforce_detection=False)[0]
# test_embedding = test_embedding_obj["embedding"]

# Similarity function
def find_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# embeddings from the PostgreSQL
conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT name, embedding FROM face_data")
embeddings = cur.fetchall()
cur.close()
conn.close()

# Path to the test image
test_img_path = "D:/Desktop/appliedml/Face Recognition Project/test.jpg"

# Generate embedding for the test image
try:
    test_embedding_obj = DeepFace.represent(
        img_path=test_img_path,
        model_name='Facenet',
        enforce_detection=False
    )[0]
    test_embedding = test_embedding_obj["embedding"]
except Exception as e:
    print(f"Error generating embedding for test image: {e}")
    exit()

# Compare to all stored embeddings
best_match = None
highest_similarity = -1
threshold = 0.75  # Adjust as needed

for name, emb in embeddings:
    sim = find_similarity(test_embedding, emb)
    print(f"Similarity with {name}: {sim:.4f}")
    if sim > highest_similarity:
        highest_similarity = sim
        best_match = name

# Show result
if highest_similarity >= threshold:
    print(f"\nBest match: {best_match} (Similarity: {highest_similarity:.4f})")
else:
    print("\nNo match found (all below threshold).")

# # Compare to all stored embeddings
# best_match = None
# highest_similarity = -1  # Start from lowest possible similarity

# for name, emb in embeddings:
#     sim = find_similarity(test_embedding, emb)
#     print(f"Similarity with {name}: {sim:.4f}")
#     if sim > highest_similarity:
#         highest_similarity = sim
#         best_match = name

# # Show result
# if best_match:
#     print(f"\nBest match: {best_match} (Similarity: {highest_similarity:.4f})")
# else:
#     print("No match found.")
