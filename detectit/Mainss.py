import cv2

face_cap = cv2.CascadeClassifier("C:/Users/nitya/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml")

print(cv2.__version__)
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read() #video cap
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)#makes the colours of video into b&w and then back again
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors= 5,
        minSize=(30,30),
        flags =cv2.CASCADE_SCALE_IMAGE
    )#it Captures the muscles of face or can say features of face
    for(x,y,w,h) in faces:
        #dimensions of rectangle
        cv2.rectangle(video_data,(x,y),(x+w, y+h),(0,255,0),2)
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("video_live", video_data)# shows the box

    if cv2.waitKey(50) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()

# print(cv2.__version__)
# video_cap = cv2.VideoCapture(0)

# while True:
#     ret, video_data = video_cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     cv2.imshow("video_live", video_data)

#     if cv2.waitKey(50) == ord("a"):
#         break

# video_cap.release()
# cv2.destroyAllWindows()
