# import the necessary packages
import numpy as np
import cv2

class CoverDescriptor:
	def __init__(self, useSIFT = False):
		# store whether or not SIFT should be used as the feature
		# detector and extractor
		self.useSIFT = useSIFT

	def describe(self, image):
		# initialize the BRISK detector and feature extractor (the
		# standard OpenCV 3 install includes BRISK by default)
		descriptor = cv2.BRISK_create()

		# check if SIFT should be utilized to detect and extract
		# features (this this will cause an error if you are using
		# OpenCV 3.0+ and do not have the `opencv_contrib` module
		# installed and use the `xfeatures2d` package)
		if self.useSIFT:
			descriptor = cv2.xfeatures2d.SIFT_create()

		# detect keypoints in the image, describing the region
		# surrounding each keypoint, then convert the keypoints
		# to a NumPy array
		(kps, descs) = descriptor.detectAndCompute(image, None)
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and descriptors
		return (kps, descs)