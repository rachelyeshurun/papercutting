import errno
import os
import sys

import numpy as np
import cv2

from glob import glob
import utils
from foreground import foreground_extractor
import threshold
import connect
# import black_and_white as bw
# import connect as conn


def runPipeline(image, threshold):
    """TODO
    """
    raise NotImplementedError


def readImages(image_dir):
    """This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
    ----------
        image_dir : str
            The image directory to get images from.

    Returns:
    ----------
        images : list
            List of images in image_dir. Each image in the list is of type
            numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
                .format(image_dir))

    return images

def nothing(x):
        pass

if __name__ == "__main__":


    image_dir = ''

    in_dir = os.path.join("source", image_dir)
    out_dir = os.path.join("output", image_dir)

    # For each image in directory called 'source':
    # Read image
    # Process into a papercut image
    # Save output to same name concatenated with '_papercut'

    print("Reading images.", image_dir)
    image_list = utils.readImages(image_dir)
    #TODO make resize flag a user parameter
    resize = False

    fg = foreground_extractor()

    # todo: should
    haar_path = 'C:\\Users\\RachelAdmin\\Anaconda2\\envs\\py36\\Library\\etc\\haarcascades'

    for img in image_list:
        image, name = img
        # Subsampling the images can reduce runtime for large files
        if image.shape[0] > 800 or image.shape[1] > 800:
            resize = True
        if resize:
            image = image[::4, ::4]
        # utils.show_image(image,name)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        # foreground = np.copy(image)
        # foreground = fg.extract(image)
        # black = np.where(foreground == 0)
        # foreground[black] = 255
        # utils.show_image(foreground, 'white background')
        # foreground = image
        # B, G, R = cv2.split(foreground)
        # foreground = np.maximum(B, G, R)
        # utils.show_image(foreground, 'max of B, G, R')

        # bin = threshold.bilateral_filter(foreground, color=50,space=3)
        # utils.show_image(bin, 'bilateral result')

        # B, G, R = cv2.split(image)
        # foreground = np.maximum(B, G, R)

        # First convert to grayscale based on luminance (not max RBG as in paper)
        # (L, A, B) = cv2.split(foreground)
        (L, A, B) = cv2.split(image)
        base = np.copy(L)
        display_image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        # utils.show_image(base, 'base image')
        utils.show_image(display_image, 'display image')

        num_faces = 0

        # Using pre-trained models in opencv to allow user to threshold facial features separately
        # Following this tutorial (with some additions): https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
        faces = face_cascade.detectMultiScale(base, 1.05, 3)

        # todo: handle more than one face in an image ...
        if len(faces) > 1:
            raise NotImplementedError


        # If no face in image, then just give the user one threshold ('outer')
        if len(faces) == 0:
            outer_mask = utils.get_mask(base, (0, 0, base.shape[1], base.shape[0]))
        else:
            num_faces = 1
            face = faces[0]
            display_image = utils.draw_rectangle_on_image(display_image, face, (255, 0, 0))
            (x, y, w, h) = face
            roi_gray = base[y:y + h, x:x + w]
            roi_color = display_image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # left/right from POV of user ..
            right_eye = (x + eyes[0][0], y + eyes[0][1], eyes[0][2], eyes[0][3])
            left_eye =(x + eyes[1][0], y + eyes[1][1], eyes[1][2], eyes[1][3])
            display_image = utils.draw_rectangle_on_image(display_image, left_eye, (255, 255, 0))
            display_image = utils.draw_rectangle_on_image(display_image, right_eye, (255, 0, 255))


        # create a list of regions to threshold and a default threshold (the median?)
        # list of tuples: (rectangle, threshold val)



        left_eye_mask = utils.get_mask(base, left_eye, circle = True)
        right_eye_mask = utils.get_mask(base, right_eye, circle= True)
        utils.show_image(left_eye_mask, 'left_eye_mask')
        # not_eyes_mask = cv2.bitwise_not(cv2.bitwise_or(left_eye_mask, right_eye_mask))
        face_mask = utils.get_mask(base, face)
        utils.show_image(face_mask, 'face mask')
        outer_mask = cv2.bitwise_not(face_mask)
        utils.show_image(outer_mask, 'outer')
        face_mask = cv2.bitwise_xor(face_mask, cv2.bitwise_or(left_eye_mask, right_eye_mask))
        utils.show_image(face_mask, 'face mask')


        # threshold_list = []
        # threshold_list.append((face_mask, utils.local_median(base, face)))
        # threshold_list.append((left_eye_mask, utils.local_median(base, left_eye)))
        # threshold_list.append((right_eye_mask, utils.local_median(base, right_eye)))


        utils.show_image(display_image, 'display image')

        def nothing(x):
            pass

        # Create a black image the size of the original
        img = np.zeros((base.shape[0], base.shape[1]), np.uint8)
        cv2.namedWindow('image')
        cv2.namedWindow('thresholded image')
        cv2.imshow('thresholded image', base)

        # create trackbars for color change
        cv2.createTrackbar('canny_lower', 'image', 50, 255, nothing)
        cv2.createTrackbar('canny_upper', 'image', 100, 255, nothing)
        cv2.createTrackbar('sigma_color', 'image', 70, 255, nothing)
        cv2.createTrackbar('sigma_space', 'image', 11, 100, nothing)

        cv2.createTrackbar('left_eye_threshold', 'image', 100, 255, nothing)
        cv2.createTrackbar('right_eye_threshold', 'image', 100, 255, nothing)
        cv2.createTrackbar('face_threshold', 'image', 100, 255, nothing)
        cv2.createTrackbar('outer_threshold', 'image', 100, 255, nothing)

        # create switch for ON/OFF functionality
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'image', 0, 1, nothing)

        while (1):
            cv2.imshow('thresholded image', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            lo = cv2.getTrackbarPos('canny_lower', 'image')
            hi = cv2.getTrackbarPos('canny_upper', 'image')
            colour = cv2.getTrackbarPos('sigma_color', 'image')
            space = cv2.getTrackbarPos('sigma_space', 'image')

            left_eye_thresh = cv2.getTrackbarPos('left_eye_threshold', 'image')
            right_eye_thresh = cv2.getTrackbarPos('right_eye_threshold', 'image')
            face_thresh = cv2.getTrackbarPos('face_threshold', 'image')
            outer_thresh = cv2.getTrackbarPos('outer_threshold', 'image')

            # todo: improve this part - shouldn't need to create the whole list each time, just update the threshold value!
            # update the face threshold vals
            threshold_list = []
            threshold_list.append((not_eyes_mask, outer_thresh))
            threshold_list.append((left_eye_mask, left_eye_thresh))
            threshold_list.append((right_eye_mask, right_eye_thresh))

            # threshold_list.append((face_mask, face_thresh))
            # threshold_list.append((outer_mask, outer_thresh))

            s = cv2.getTrackbarPos(switch, 'image')

            if s == 0:
                img[:] = 0
            else:
                img[:] = threshold.gray2binary(base, canny_lower=lo, canny_upper=hi, sigma_color=colour, sigma_space = space, thresholds = threshold_list)

        # user pressed esc. Now enforce connectivity
        cv2.destroyAllWindows()

        # Warning: hack ahead! Doing some morphological close to get rid of 1 pixel holes which currently make the next step crash
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        conn = connect.ComponentConnector(img, L)
        connected_image = conn.connect_and_thicken()
        utils.show_image(connected_image, 'result')

        # morphological close to fill in any tiny holes leftover (note that 'holes' for us are white, but
        # holes for cv2.dilate are black - so invert before and after dilate)
        kernel = np.ones((2, 2), np.uint8)
        connected_image = np.bitwise_not(cv2.dilate(np.bitwise_not(connected_image), kernel, iterations=2))
        utils.show_image(connected_image, 'result')





    exit(0)

 # papercut_image = runPipeline(images, threshold=100)


# TODO - convert to vector graphics - this might be a manual step

# TODO - save image
