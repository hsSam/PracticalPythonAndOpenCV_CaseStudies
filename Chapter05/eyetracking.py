from utilities.eyetracker import EyeTracker
from utilities import imutils
import cv2

# Define paths
video_path = 'video/adrian_eyes.mov'
face_cascade_path = 'cascades/haarcascade_frontalface_default.xml'
eye_cascade_path = 'cascades/haarcascade_eye.xml'

# construct the eye tracker
eye_tracker = EyeTracker(face_cascade_path, eye_cascade_path)

# Load the video
camera = cv2.VideoCapture(video_path)

while True:
    # Grab the current frame
    (ok, frame) = camera.read()

    # Check if at the end of the video
    if not ok:
        break

    # Resize the frame and convert it to greyscale
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes in the image
    boxes = eye_tracker.track(gray)

    # Draw the face bounding boxes
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Show the tracked eyes and face
    cv2.imshow("Tracking", frame)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
