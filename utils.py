import errno
import os
import sys

import numpy as np
import cv2
from glob import glob

from matplotlib import pyplot as plt

#Based on code from main.py given to us for assignment A7 Video Textures

def show_image(image, title=None):
    if True:
        image_copy = np.copy(image)

        if len(image.shape) < 3:
            colours = "gray"
        else:
            colours = "viridis"
        plt.imshow(image_copy, cmap=colours)
        plt.title(title)
        plt.colorbar()
        plt.show()

def readImages(self, image_dir = "source"):
    """This function reads in input images from a given image directory.
    Args:
    ----------
        image_dir : str
            The image directory to get images from.
    Returns:
    ----------
        images : list of tuple <image base name, image>
            List of images in image_dir. Each image in the list is of type numpy.ndarray.
    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))

    names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
                .format(self.in_dir))

    return list(zip(images, names))


def save_image(self, image, name = "out",  output_dir = "output"):
    """Convenient wrapper for writing images to the output directory."""
    raise NotImplementedError
    # if True:
    #     filename = datetime.utcnow().strftime('%Y%m%d-%H-%M-%S-%f')[:-3] + '.png'
    #     image = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #     cv2.imwrite(os.path.join(OUT_FOLDER, filename), image)
    #
    # tmp_errors=[]
    # for f in range(self.Xtrain.shape[1]):
    #     tmp_result = self.weights*(tmp_signs[f]*((self.Xtrain[:,f]>tmp_thresholds[f])*2-1) != self.ytrain)
    #     tmp_errors.append(sum(tmp_result))
    #
    #
    # feat = tmp_errors.index(min(tmp_errors))
    #
    # self.feature = feat
    # self.threshold = tmp_thresholds[feat]
    # self.sign = tmp_signs[feat]
    # # -- print self.feature, self.threshold
    # # print "self.feature, self.threshold", self.feature, self.threshold

def draw_rectangle_on_image(image, rectangle, colour = (255, 0, 0)):
    x, y, w, h = rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
    return image

def local_median(image, rectangle):
    ''' get median within the given rectangle of the image'''
    x, y, w, h = rectangle
    return np.median(image[y:y+h, x:x+w])

def get_mask(image, rectangle, circle = False):
    ''' get median within the given rectangle of the image'''
    x, y, w, h = rectangle
    mask = np.zeros_like(image)
    if not circle:
        mask[y:y+h, x:x+w] = 255
    else:
        # https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python
        L = [[x,y], [x + w, y], [x, y + h], [x + w, y + h]]
        ctr = np.array(L).reshape((-1, 1, 2)).astype(np.int32)
        (cx, cy), r = cv2.minEnclosingCircle(ctr)
        center = ( int(cx), int(cy) )
        cv2.circle(mask, center,int(r), (255, 255, 255), -1)
    return mask
