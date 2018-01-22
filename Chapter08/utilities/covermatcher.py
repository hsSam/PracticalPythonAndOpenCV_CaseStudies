import numpy as np
import cv2


class CoverMatcher:
	def __init__(self, descriptor, cover_paths, ratio=0.7, min_matches=40, use_hamming=True):
		# Store the descriptor, book cover paths, ratio and minimum number of matches for the homography calculation,
		# then initialize the distance metric to be used when computing the distance between features
		self.descriptor = descriptor
		self.cover_paths = cover_paths
		self.ratio = ratio
		self.min_matches = min_matches
		self.distance_method = "BruteForce"

		# If the Hamming distance should be used, then update the distance method
		if use_hamming:
			self.distance_method += "-Hamming"

	def search(self, query_keypoints, query_descriptors):
		# Initialize the dictionary of results
		results = {}

		# Loop over the book cover images
		for path in self.cover_paths:
			# Load the query image, convert it to greyscale, and extract keypoints and descriptors
			cover = cv2.imread(path)
			gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
			(keypoints, descriptors) = self.descriptor.describe(gray)

			# Determine the number of matched, inlier keypoints, and update the results
			score = self.match(query_keypoints, query_descriptors, keypoints, descriptors)
			results[path] = score

		# If matches were found, sort them
		if len(results) > 0:
			results = sorted([(v, k) for (k, v) in results.items() if v > 0], reverse=True)

		# Return the results
		return results

	def match(self, key_a, fea_a, key_b, fea_b):
		# Compute the raw matches and initialize the list of actual matches
		matcher = cv2.DescriptorMatcher_create(self.distance_method)
		raw_matches = matcher.knnMatch(fea_b, fea_a, 2)
		matches = []

		# Loop over the raw matches
		for match in raw_matches:
			# Ensure the distance is within a certain ratio of each other
			if len(match) == 2 and match[0].distance < match[1].distance * self.ratio:
				matches.append((match[0].trainIdx, match[0].queryIdx))

		# Check to see if there are enough matches to process
		if len(matches) > self.min_matches:
			# Construct the two sets of points
			poi_a = np.float32([key_a[i] for (i, _) in matches])
			poi_b = np.float32([key_b[j] for (_, j) in matches])

			# Compute the homography between the two sets of points and compute the ratio of matched points
			(_, status) = cv2.findHomography(poi_a, poi_b, cv2.RANSAC, 4.0)

			# Return the ratio of the number of matched keypoints to the total number of keypoints
			return float(status.sum()) / status.size

		# No matches were found
		return -1.0
