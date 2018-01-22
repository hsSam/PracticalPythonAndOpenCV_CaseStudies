NOTES ON PARAMETERS:

scaleFactor: How much the image size is reduced at each image scale. This
value is used to create the scale pyramid. In order to detect faces at
multiple scales in the image (some faces may be closer to the foreground,
and thus be larger, other faces may be smaller and in the background, thus
the usage of varying scales). A value of 1.05 indicates you are reducing
the size of the image by 5% at each level in the pyramid.

minNeighbors: How many neighbors each bounding box should have for the area
in the bounding box to be considered a face. The cascade classifier will
detect multiple bounding boxes around a face. This parameter controls how
many rectangles (neighbors) need to be detected for the box to be labeled
a face.

minSize: A tuple of width and height indicating the minimum size of the
bounding box. Bounding boxes smaller than this size are ignored. It is a
good idea to start with (30, 30) and go from there.

maxSize: A tuple of width and height indicating the maximum size of the
bounding box. Bounding boxes larger than this size are ignored. This parameter
is not usually manually set.