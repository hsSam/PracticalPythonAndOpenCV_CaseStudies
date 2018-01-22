from utilities.facedetector import FaceDetector
import cv2

# Define paths
image_path = 'images/obama.png'
cascade_path = 'cascades/haarcascade_frontalface_default.xml'

# Load the image and convert it to greyscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find faces in the image
detector = FaceDetector(cascade_path)
face_boxes = detector.detect(gray, 1.2, 5)
print("{} face(s) found".format(len(face_boxes)))

# Loop over the faces and draw a rectangle around each
for (x, y, w, h) in face_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
