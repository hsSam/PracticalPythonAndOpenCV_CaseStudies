from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from utilities.hog import HOG
from utilities import dataset


# Define paths
dataset_path = 'data/digits.csv'
model_path = 'models/svm.cpickle'

# Load the dataset and initialize the data matrix
(digits, target) = dataset.load_digits(dataset_path)
data = []

# Initialize the HOG descriptor
hog = HOG(orientations=18, pixels_per_cell=(10, 10), cells_per_block=(1, 1), transform=True)

# Loop over the images
for image in digits:
	# De-skew the image and center it
	image = dataset.de_skew(image, 20)
	image = dataset.center_extent(image, (20, 20))

	# Describe the image and update the data matrix
	hist = hog.describe(image)
	data.append(hist)

# Train the model
model = LinearSVC(random_state=42)
model.fit(data, target)

# Save the model to file
joblib.dump(model, model_path)
