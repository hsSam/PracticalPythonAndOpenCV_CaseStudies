from skimage import feature


class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), transform=False):
        # Save the number of orientations, pixels per cell, cells per block, and whether or not power law compression
		# should be applied
        self.orienations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform = transform

    def describe(self, image):
        # Compute HOG for the image
        hist = feature.hog(image, orientations=self.orienations, pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block, transform_sqrt=self.transform)

        # Return the HOG features
        return hist
