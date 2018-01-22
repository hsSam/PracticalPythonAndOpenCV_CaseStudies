from sklearn.externals import joblib
from utilities.hog import HOG
from utilities import dataset
import mahotas
import cv2


# Define paths
model_path = 'models/svm.cpickle'
image_path = 'images/digit_sample.png'

# Load the model
model = joblib.load(model_path)

# Initialize the HOG descriptor
hog = HOG(orientations=18, pixels_per_cell=(10, 10), cells_per_block=(1, 1), transform=True)

# Load the image and convert it to greyscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image, find edges, and find contours along the edged regions
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
(_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by their x-axis position, ensuring that we read the numbers from left to right
contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

# Loop over the contours
for (contour, _) in contours:
	# Compute the bounding box for the rectangle
	(x, y, w, h) = cv2.boundingRect(contour)

	# If the width is at least 7 pixels and the height is at least 20 pixels, the contour is likely a digit
	if w >= 7 and h >= 20:
		# Crop the ROI and then threshold the greyscale ROI to reveal the digit
		roi = gray[y:y + h, x:x + w]
		thresh = roi.copy()
		t = mahotas.thresholding.otsu(roi)
		thresh[thresh > t] = 255
		thresh = cv2.bitwise_not(thresh)

		# De-skew the image center its extent
		thresh = dataset.de_skew(thresh, 20)
		thresh = dataset.center_extent(thresh, (20, 20))

		cv2.imshow("thresh", thresh)

		# Extract features from the image and classify it
		hist = hog.describe(thresh)
		digit = model.predict([hist])[0]
		print("Prediction: {}".format(digit))

		# Draw a rectangle around the digit, the show what the digit was classified as
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(image, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
		cv2.imshow("image", image)
		cv2.waitKey(0)
