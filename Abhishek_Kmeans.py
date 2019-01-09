# USAGE
# python Abhishek_Kmeans.py --image images/2.jpg --clusters 3

# import the necessary packages

import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

def EucDist(pixel1 , pixel2):                               # calc dist bw two points
	dist = np.sum((pixel1 - pixel2)**2)
	return dist

def InitRandCentroids(pixel_array,k):   					# Take pixel array and no. of clusters
	index_array = np.arange(np.shape(pixel_array)[0])		# to randomly init centroids
	np.random.shuffle(index_array)
	#rand = np.random.randint(k,size)   					# returns a random list in range from  0 to len(pixel)-1
	init_centroids = np.array(pixel_array[index_array[:k],:])
	return init_centroids

def assignLabels(centroid_array, pixel_array):
	pixel_labels = np.empty((0))
	pixel_index = 0
	for pixel in pixel_array:
		centroid_index = 0
		dist_pixel_centroid = np.empty((0))
		for centroid in centroid_array:
			dist_pixel_centroid = np.append(dist_pixel_centroid,EucDist(pixel, centroid))  # for each pixel its dist is calc from each centroid and whichever centroid is closest from it the pixel is assigned that label
		#	print("e")
			centroid_index = centroid_index + 1
		#	print("f")
		pixel_labels = np.append(pixel_labels,np.argmin(dist_pixel_centroid))
		#print("g")
		pixel_index = pixel_index + 1
		#print("h")
	return pixel_labels


def updateCentroids(pixel_array,pixel_labels,old_centroids):
	new_centroids = np.empty((0,3))
	for i in range(0,np.shape(old_centroids)[0]):
		mask = np.array(pixel_labels == i) 
		masked_array = pixel_array[mask,:]
		new_centroids = np.vstack([new_centroids,np.mean(masked_array, axis = 0)])    # centroids are updated based on the mean of those pixels having same label
	return new_centroids

def kMeans(X,iters,n_clusters):
	centroids = InitRandCentroids(X,n_clusters)
	print("initial centroid assigning done!")
	pixels = X
	i = 0
	for iteration in range(0,iters):
		i = i + 1
		print('Iteration:' + str(i))
		pixelLabels = assignLabels(centroids,pixels)
		print("Label assigning done")
		centroids = updateCentroids(pixels,pixelLabels,centroids)
		print("centroid update done")
		print(centroids)
	return pixelLabels,centroids

def centroid_histogram(labels):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(labels)) + 1)
	(hist, _) = np.histogram(labels, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar


'''# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
'''

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread('new.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

labels,centroid_list = kMeans(np.array(image), iters = 5, n_clusters = 2)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = Abhishek_utils.centroid_histogram(labels)
bar = Abhishek_utils.plot_colors(hist, centroid_list)

hist = centroid_histogram(labels)
bar = plot_colors(hist, centroid_list)
centroid_list
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
