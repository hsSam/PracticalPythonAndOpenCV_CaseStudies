# USAGE
# python search.py --db books.csv --covers covers --query queries/query01.png

# import the necessary packages
from __future__ import print_function
from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatcher import CoverMatcher
import argparse
import glob
import csv
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True,
	help = "path to the book database")
ap.add_argument("-c", "--covers", required = True,
	help = "path to the directory that contains our book covers")
ap.add_argument("-q", "--query", required = True,
	help = "path to the query book cover")
ap.add_argument("-s", "--sift", type = int, default = 0,
	help = "whether or not SIFT should be used")
args = vars(ap.parse_args())

# initialize the database dictionary of covers
db = {}

# loop over the database
for l in csv.reader(open(args["db"])):
	# update the database using the image ID as the key
	db[l[0]] = l[1:]

# initialize the default parameters using BRISK is being used
useSIFT = args["sift"] > 0
useHamming = args["sift"] == 0
ratio = 0.7
minMatches = 40

# if SIFT is to be used, then update the parameters
if useSIFT:
	minMatches = 50

# initialize the cover descriptor and cover matcher
cd = CoverDescriptor(useSIFT = useSIFT)
cv = CoverMatcher(cd, glob.glob(args["covers"] + "/*.png"),
	ratio = ratio, minMatches = minMatches, useHamming = useHamming)

# load the query image, convert it to grayscale, and extract
# keypoints and descriptors
queryImage = cv2.imread(args["query"])
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)

# try to match the book cover to a known database of images
results = cv.search(queryKps, queryDescs)

# show the query cover
cv2.imshow("Query", queryImage)

# check to see if no results were found
if len(results) == 0:
	print("I could not find a match for that cover!")
	cv2.waitKey(0)

# otherwise, matches were found
else:
	# loop over the results
	for (i, (score, coverPath)) in enumerate(results):
		# grab the book information
		(author, title) = db[coverPath[coverPath.rfind("/") + 1:]]
		print("{}. {:.2f}% : {} - {}".format(i + 1, score * 100,
			author, title))

		# load the result image and show it
		result = cv2.imread(coverPath)
		cv2.imshow("Result", result)
		cv2.waitKey(0)