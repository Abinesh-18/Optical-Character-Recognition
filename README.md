# Optical-Character-Recognition
The goal of this task is to implement an optical character recognition system. Experiment with connected component and matching algorithms and the goal is both detection and recognition of various characters.
# INTRODUCTION:
In this project, Optical Character Recognition (OCR) is performed, using the pixel coordinates feature
vector from both the template and the testing image and compared using Sum of Squared
Differences (SSD).
# KEY Concepts used:
#  Connected Component Labeling (CCL)
Connected components labeling((CCL) scans the image and groups its pixels into
components based on pixel connectivity, i.e. all pixels in a connected component have same
pixel intensity values and are in some way connected with each other. After all groups have
been determined, each pixel is labeled with a number of ascending order or can also be done
by colors. In this code they are labeled with numbers in ascending order.
It is of 2 types:
i) 4-connectivity:
ii) 8-connectivity:
In the case of 4-connectivity, only the pixels directly above, below, right and left are counted
as neighbors, whereas in the case of 8-connectivity the four adjacent diagonal pixels are
counted as neighbors
It is a two pass algorithm
In the first pass we scan through the image, pixel by pixel, and look at each pixel's neighbors.
A pixel's neighbors are the pixels that immediately surround it.(Only the top and left pixel
are considered)
(1) If the pixel has no labelled neighbors (no neighboring foreground pixels). It should
be given a new label
(2) The pixel has one or more neighbors with the same label. The current pixel is
therefore part of the same shape, so is given the same label as its neighbor(s).
The pixel is given the least value among its neighbors
In the second pass:
(1) Again scanning through the image pixel by pixel, for each labelled pixel we check
if we recorded any equivalent labels in our disjoint-set data structure. If we did,
then we replace the pixel's label with the lowest label in its equivalence set.
Note; This Algorithm fails at certain instances (fails for characters such as i,j )
#  Sum of squared Differences
Sum of squared differences (SSD) is one of measure of match that based on pixel by pixel
intensity differences between the two images.
It is calculated using the formula:
R(x,y)=Σx′,y′(T(x′,y′)−I(x+x′,y+y′))2
T - Template I - Image
# Computed Features:
The pixel coordinates are taken as the features for this project. The feature considered here, is
extracted for both sets of images and compared using SSD.
# Detection:
The detection is done using 4-connectivity Connected Component labeling (CCL) Algorithm.
The steps are as follows:
1. The given test image which consists of sentences needs to be split into components of
individual characters. This is done using 4-connectivity 2 pass Connected component labeling
(CCL) Algorithm.
2. After CCL the characters are labeled with numbers from 1 to endof(characters)
3. The coordinates and dimensions of the labeled characters are calculated
4. The labels are cropped from the given image and stored in individual lists which consists of
the coordinates, dimensions, and the cropped image matrix
5. This is passed to the ocr() function
# Recognition:
1. Recognition is done by passing the Template list from enrollment(It contains information
about the template images) and passing the Label List from Detection (It contains
information about the testing Image)
2. Once passed to the function the value of SSD is calculated by pixel-pixel intensity comparison.
Comparing the SSD values calculated for every template select the minimum value among
them. Compare the minimum value with the threshold value set, to determine whether the
character matches or not.
