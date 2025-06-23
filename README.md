Face Recognition System with DeepFace and PostgreSQL

This project is a real-time face recognition system using a webcam. 

It leverages DeepFace for face embeddings and PostgreSQL for storing and retrieving user data.

Current Features:

-Register users via webcam and store their face embeddings in a PostgreSQL database.

-Recognize users in real-time via webcam using cosine similarity on embeddings.

-Easily extendable for future features like image-vs-image recognition.

Project Structure:

├── main.py               # Real-time face recognition via webcam

├── register_user.py      # Register a user using webcam and store embeddings

├── db_utils.py           # PostgreSQL connection setup

├── stored-faces/         # (Optional) Folder to save cropped face images

├── README.md             # Project documentation
