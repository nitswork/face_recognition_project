import pickle
from db_utils import get_connection

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

conn = get_connection()
cur = conn.cursor()

for filename, embedding in embeddings:
    cur.execute("INSERT INTO face_data (filename, embedding) VALUES (%s, %s)", (filename, embedding))

conn.commit()
cur.close()
conn.close()
