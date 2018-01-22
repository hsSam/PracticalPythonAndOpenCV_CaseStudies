import cv2


class EyeTracker:
    def __init__(self, face_cascade_path, eye_cascade_path):
        # load the face and eye detector
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def track(self, image):
        # Detect faces in the image and initialize the list of rectangles containing the faces and eyes
        face_boxes = self.face_cascade.detectMultiScale(image, 1.1, 5)
        boxes = []

        # Loop over the face bounding boxes
        for (f_x, f_y, f_w, f_h) in face_boxes:
            # Extract the face ROI and update the list of bounding boxes
            face_roi = image[f_y:f_y + f_h, f_x:f_x + f_w]
            boxes.append((f_x, f_y, f_x + f_w, f_y + f_h))

            # Detect eyes in the face ROI
            eye_boxes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 10)

            # Loop over the eye bounding boxes
            for (e_x, e_y, e_w, e_h) in eye_boxes:
                # Update the list of bounding boxes
                boxes.append((f_x + e_x, f_y + e_y, f_x + e_x + e_w, f_y + e_y + e_h))

        # Return the bounding boxes around the faces and eyes
        return boxes
