''' connect all the black components of the image and thicken the connecting paths
'''
import errno
import os
import sys

import numpy as np
import cv2

from utils import show_image

import numpy as np
import cv2

from pprint import pprint

# The following class is based on answer by Boaz Yaniv here: https://stackoverflow.com/questions/5997189/how-can-i-make-a-unique-value-priority-queue-in-python
# I need to wrap the simple heap q with a set so that I can update priority. So we implement a 'priority set'
import heapq
class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, d, pri):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)
        else:
            # if pixel is already in the unvisited queue, just update its priority
            prevpri, d = heapq.heappop(self.heap)
            heapq.heappush(self.heap, (pri, d))

    def get(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return d, pri

    def is_empty(self):
        return len(self.set) == 0

    def print(self):
        print ('currently unvisited pixels with their priority (distance):')
        pprint(self.heap)


# Using some of this code as a basis, but need to change a lot of things
# http://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php
        # Create 'distance' (sum of intensities) image - inf
        # Create 'prev' image (to get path)
        # Find the smallest component, store the label number of the smallest component - call it the current component
        # Get border pixels of current component
        # Create a list of tuples: (coordinates
        # Heapify border pixels (this is the Q) with infinite distance
        # heapq to store unvisited vertices (pixels) with their distance (sum of intensities).
        # while Q is not empty
        #   extract the minimum distance unvisited pixel (call it the 'current pixel')
        #   for each of the 8 neighboring pixels:
        #       if neigbouur belongs to another component - stop algorithm! we found the path.
        #       if neighbour belongs to current component - skip
        #       calculate the neighbour's distance from the current pixel by adding the current pixel's distance with the neighbour's intensity value
        #           if the neighbour's calculated distance is less than its current distance, then update its distance.
        #           also add to the priority set (update if already there)

class ComponentConnector(object):
    def __init__(self, binary_image, intensity_image):
        self.current_image = binary_image
        self.intensities = intensity_image
        self.rows = binary_image.shape[0]
        self.cols = binary_image.shape[1]
        self.distances = np.ndarray((self.rows, self.cols, 2), dtype = np.uint64)
        self.prev_array = np.ndarray((self.rows, self.cols, 2), dtype = np.uint64)
        self.current_component_mask = np.zeros_like(intensity_image)
        self.other_component_mask = np.zeros_like(intensity_image)
        self.unvisited = PrioritySet()
        self.path_end = (0,0)

    def connect_and_thicken(self):
        num_components = self.connect_smallest_component()
        while num_components > 1:
            num_components = self.connect_smallest_component()
        return self.current_image

    def connect_smallest_component(self):
        # inspired by https://stackoverflow.com/questions/47055771/how-to-extract-the-largest-connected-component-using-opencv-and-python

        # initialize the distance and prev_array (for the path)
        self.distances = np.ones((self.rows, self.cols), np.uint64) * sys.maxsize
        self.prev_array = np.zeros((self.rows, self.cols, 2), dtype = np.uint64)

        while True:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(self.current_image),
                                                                                    connectivity=8, ltype=cv2.CV_32S)
            print('num_labels\n', num_labels)
            # print('labels\n', labels, labels.shape)
            print('stats\n', stats, stats.shape)
            # print('centroids\n', centroids, centroids.shape)

            # get label of smallest componenent (note - the first stat is the background, so look at rows starting from 1)
            smallest_label = 1 + np.argmin(stats[1:, cv2.CC_STAT_AREA])
            print('smallest label: ', smallest_label)
            area = stats[smallest_label, cv2.CC_STAT_AREA]
            x = stats[smallest_label, cv2.CC_STAT_LEFT]
            y = stats[smallest_label, cv2.CC_STAT_TOP]
            if area == 1:
                # fill in any components with area 1
                self.current_image[y, x] = 0
                show_image(self.current_image, 'filled hole?')
            else:
                break

        if num_labels < 3:
            print('done')
            return 1

        # get border pixels of smallest label -
        self.other_component_mask = np.zeros_like(self.intensities)
        self.other_component_mask[np.logical_and(labels != smallest_label, labels != 0)] = 255
        self.current_component_mask = np.zeros_like(self.intensities)
        self.current_component_mask[labels == smallest_label] = 255

        # show_image(self.other_component_mask, 'other_components_mask')
        # show_image(self.current_component_mask, 'current_components_mask')

        contourImg, contours, hierarchy = cv2.findContours(self.current_component_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # mask = np.zeros_like(bin_inverted)
        # cv2.drawContours(mask, contours, -1, (255), 1)
        # show_image(mask, 'contourImg')

        # add the component border pixels to the unvisited set of vertices (pixels)
        # todo: get a run time error here if the area is 1 pixel ..
        borderList = [tuple(c) for c in np.vstack(contours).squeeze()]
        self.unvisited = PrioritySet()
        for coord in borderList:
            self.unvisited.add((coord[1], coord[0]), self.intensities[coord[1], coord[0]])
        self.unvisited.print()

        while not self.unvisited.is_empty():
            current_pixel, distance = self.unvisited.get()
            row, col = current_pixel  # coordinates from contour are col, row not row, col ...
            # print('current pixel (row, col, distance):', row, col, distance)
            # check out the neighbours
            if row > 0 and col > 0:
                other_component = self.evaluate_neighbour(row, col, distance, row - 1, col - 1)
                if other_component:
                    break
            if row > 0:
                other_component = self.evaluate_neighbour(row, col, distance,  row - 1, col)
                if other_component:
                    break
            if row > 0 and col < self.cols - 1:
                other_component = self.evaluate_neighbour(row, col, distance,  row - 1, col + 1)
                if other_component:
                    break
            if col < self.cols - 1:
                other_component = self.evaluate_neighbour(row, col, distance,  row, col + 1)
                if other_component:
                    break
            if row < self.rows - 1 and col < self.cols - 1:
                other_component = self.evaluate_neighbour(row, col, distance,  row + 1, col + 1)
                if other_component:
                    break
            if row < self.rows - 1:
                other_component = self.evaluate_neighbour(row, col, distance,  row + 1, col)
                if other_component:
                    break
            if row < self.rows - 1 and col > 0:
                other_component = self.evaluate_neighbour(row, col, distance,  row + 1, col - 1)
                if other_component:
                    break
            if col > 0:
                other_component = self.evaluate_neighbour(row, col, distance,  row, col - 1)
                if other_component:
                    break

        r, c = self.path_end
        print ('path end:', r, c)
        # print ('prev array', self.prev_array)
        # todo: colour the path pixels in black (update the current image)
        while not self.current_component_mask[r, c]:
            # print('current comp mask', self.current_component_mask[r, c])
            self.current_image[r, c] = 0
            r, c = self.prev_array[r, c]
            # print('prev (row, col)', r, c)

        # show_image(self.current_image, 'updated with path')

        print ('prev array', self.prev_array)
        # todo: thicken the path .. (update the current image)

        return num_labels - 2

    def evaluate_neighbour(self, row, col, distance, neighbour_row, neighbour_col):

        # print('eval neighbour', row,col, neighbour_row, neighbour_col)
        if self.current_component_mask[neighbour_row, neighbour_col]:
            # print('not a neighbour.. on same component', neighbour_row, neighbour_col)
            return False
        if self.other_component_mask[neighbour_row, neighbour_col]:
            print('reached another component!!', neighbour_row, neighbour_col)
            self.path_end = (row, col)
            return True

        # print('distance: ', distance, self.intensities[neighbour_row, neighbour_col])
        sum = distance + np.uint64(self.intensities[neighbour_row, neighbour_col])
        # print ('sum, current distance of neighbour', sum, self.distances[neighbour_row, neighbour_col])
        if sum < self.distances[neighbour_row, neighbour_col]:
            self.distances[neighbour_row, neighbour_col] = sum
            # print('adding row, col to path', row, col)
            self.prev_array[neighbour_row, neighbour_col] = (row, col)
            self.unvisited.add((neighbour_row, neighbour_col), sum)
            # print('updated path', neighbour_row, neighbour_col, '<--:', self.prev_array[neighbour_row, neighbour_col])
        return False
