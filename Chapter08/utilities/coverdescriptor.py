import numpy as np
import cv2


class CoverDescriptor:
	def __init__(self, use_sift=False):
		# Store whether or not SIFT should be used as the feature detector and extractor
		self.use_sift = use_sift

	def describe(self, image):
		# Initialize the BRISK detector and feature extractor
		descriptor = cv2.BRISK_create()

		# Check if SIFT should be utilized to detect and extract features
		if self.use_sift:
			descriptor = cv2.xfeatures2d.SIFT_create()

		# Detect keypoints in the image, describing the region surrounding each keypoint, then convert the keypoints
		# to a NumPy array
		(keypoints, descriptors) = descriptor.detectAndCompute(image, None)
		keypoints = np.float32([keypoint.pt for keypoint in keypoints])

		# Return a tuple of keypoints and descriptors
		return (keypoints, descriptors)
