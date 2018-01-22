from utilities.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import glob
import cv2


# Define paths
path_to_images = 'dataset/images'
path_to_masks = 'dataset/masks'

# Grab the image and mask paths
image_paths = sorted(glob.glob(path_to_images + "/*.png"))
mask_paths = sorted(glob.glob(path_to_masks + "/*.png"))

# Initialize the list of data and class label targets
data = []
target = []

# Initialize the image descriptor
descriptor = RGBHistogram([8, 8, 8])

# Loop over the image and mask paths
for (image_path, mask_path) in zip(image_paths, mask_paths):
	# Load the image and mask
	image = cv2.imread(image_path)
	mask = cv2.imread(mask_path)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# Describe the image
	features = descriptor.describe(image, mask)

	# Update the list of data and targets
	data.append(features)
	target.append(image_path.split("_")[-2])

# Grab the unique target names and encode the labels
target_names = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# Construct the training and testing splits
(train_x, test_x, train_y, test_y) = train_test_split(data, target, test_size=0.3, random_state=42)

# Train the classifier
model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(train_x, train_y)

# Evaluate the classifier
print(classification_report(test_y, model.predict(test_x), target_names=target_names))

# Loop over a sample of the images
for i in np.random.choice(np.arange(0, len(image_paths)), 10):
	# Grab the image and mask paths
	image_path = image_paths[i]
	mask_path = mask_paths[i]

	# Load the image and mask
	image = cv2.imread(image_path)
	mask = cv2.imread(mask_path)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# Describe the image
	features = descriptor.describe(image, mask)

	# Predict what type of flower the image is
	flower = le.inverse_transform(model.predict([features]))[0]
	print(image_path)
	print("Prediction: {}".format(flower.upper()))
	cv2.imshow("image", image)
	cv2.waitKey(0)
