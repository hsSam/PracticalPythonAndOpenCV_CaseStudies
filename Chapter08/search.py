from utilities.coverdescriptor import CoverDescriptor
from utilities.covermatcher import CoverMatcher
import glob
import csv
import cv2
import ntpath


# Define paths
database_path = 'books.csv'
covers_path = 'covers'
query_path = 'queries/query01.png'

# Initialize the default parameters using BRISK is being used
use_sift = False
use_hamming = True
ratio = 0.7
min_matches = 40

# Initialize the database dictionary of covers
database = {}

# Loop over the database
for l in csv.reader(open(database_path)):
	# Update the database using the image ID as the key
	database[l[0]] = l[1:]

# If SIFT is to be used, then update the parameters
if use_sift:
	min_matches = 50

# Initialize the cover descriptor and cover matcher
cover_descriptor = CoverDescriptor(use_sift=use_sift)
cover_matcher = CoverMatcher(cover_descriptor, glob.glob(covers_path + "/*.png"), ratio=ratio, min_matches=min_matches,
							 use_hamming=use_hamming)

# Load the query image, convert it to greyscale, and extract keypoints and descriptors
query_image = cv2.imread(query_path)
gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
(query_keypoints, query_descriptors) = cover_descriptor.describe(gray)

# Try to match the book cover to a known database of images
results = cover_matcher.search(query_keypoints, query_descriptors)

# Show the query cover
cv2.imshow("Query", query_image)

# Check to see if no results were found
if len(results) == 0:
	print("A match could not be found for that cover!")
	cv2.waitKey(0)
# Otherwise, matches were found
else:
	# Loop over the results
	for (i, (score, cover_path)) in enumerate(results):
		# Grab the book information
		(author, title) = database[ntpath.basename(cover_path)]
		print("{}. {:.2f}% : {} - {}".format(i+1, score*100, author, title))

		# Load the result image and show it
		result = cv2.imread(cover_path)
		cv2.imshow("Result", result)
		cv2.waitKey(0)
