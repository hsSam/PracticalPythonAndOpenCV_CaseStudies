from utilities.facedetector import FaceDetector
from utilities import imutils
import cv2

# Define paths
video_path = 'video/adrian_face.mov'
cascade_path = 'cascades/haarcascade_frontalface_default.xml'

# Construct the face detector
detector = FaceDetector(cascade_path)

# Load the video
camera = cv2.VideoCapture(video_path)

while True:
    # Grab the current frame
    (ok, frame) = camera.read()

    # If a frame does not exist, video is over
    if not ok:
        break

    # Resize the frame and convert it to greyscale
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image and clone the frame
    face_boxes = detector.detect(gray, 1.1, 5)
    clone = frame.copy()

    # Draw the face bounding boxes
    for (x, y, w, h) in face_boxes:
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show our detected faces
    cv2.imshow("Face", clone)

    # Ff the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
