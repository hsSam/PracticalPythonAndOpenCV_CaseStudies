import cv2


class FaceDetector:
    def __init__(self, face_cascade_path):
        # Load the face detector
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect(self, image, scale_factor=1.1, min_neighbors=5):
        # Detect faces in the image
        boxes = self.face_cascade.detectMultiScale(image, scale_factor, min_neighbors, flags=cv2.CASCADE_SCALE_IMAGE)

        # Return the bounding boxes
        return boxes
