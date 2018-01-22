# import the necessary packages
import cv2

class EyeTracker:
	def __init__(self, faceCascadePath, eyeCascadePath):
		# load the face and eye detector
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
		self.eyeCascade = cv2.CascadeClassifier(eyeCascadePath)

	def track(self, image):
		# detect faces in the image and initialize the list of
		# rectangles containing the faces and eyes
		faceRects = self.faceCascade.detectMultiScale(image,
			scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE)
		rects = []

		# loop over the face bounding boxes
		for (fX, fY, fW, fH) in faceRects:
			# extract the face ROI and update the list of
			# bounding boxes
			faceROI = image[fY:fY + fH, fX:fX + fW]
			rects.append((fX, fY, fX + fW, fY + fH))
			
			# detect eyes in the face ROI
			eyeRects = self.eyeCascade.detectMultiScale(faceROI,
				scaleFactor = 1.1, minNeighbors = 10, minSize = (20, 20),
				flags = cv2.CASCADE_SCALE_IMAGE)

			# loop over the eye bounding boxes
			for (eX, eY, eW, eH) in eyeRects:
				# update the list of boounding boxes
				rects.append(
					(fX + eX, fY + eY, fX + eX + eW, fY + eY + eH))

		# return the rectangles representing bounding
		# boxes around the faces and eyes
		return rects