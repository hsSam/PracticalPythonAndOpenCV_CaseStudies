import numpy as np
import cv2

# Define path to video
video_path = 'video/iphonecase.mov'

# Define the upper and lower boundaries the colour 'blue'
blue_lower = np.array([100, 67, 0], dtype="uint8")
blue_upper = np.array([255, 128, 50], dtype="uint8")

# Load the video
camera = cv2.VideoCapture(video_path)

while True:
    # Grab the current frame
    (ok, frame) = camera.read()

    # Check to see if at the end of the video
    if not ok:
        break

    # Determine which pixels fall within the blue boundaries and blur the binary image
    blue = cv2.inRange(frame, blue_lower, blue_upper)
    blue = cv2.GaussianBlur(blue, (3, 3), 0)

    # Find contours in the image
    (_, contours, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check to see if any contours were found
    if len(contours) > 0:
        # Sort the contours and find the largest one (area)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Compute the minimum bounding box around then contour and then draw it
        box = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

    # Show the frame and the binary image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Binary", blue)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
