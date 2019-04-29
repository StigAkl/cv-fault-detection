import cv2
import numpy as np;
from matplotlib import pyplot as plt

def grayscale_to_binary(img):
    #Resize image
    resized_image = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    blur = cv2.GaussianBlur(resized_image,(5,5),0)
    ret3,th3 = cv2.threshold(blur,126,256,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
    ret, gray = cv2.threshold(resized_image,105,256,cv2.THRESH_BINARY)

    #Resource: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html


    #Add median blur
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(th3,-1,kernel)

    median_blur(gray)

def canny_edge(img):
    return cv2.Canny(img, 0,0)

def median_blur(img):
    kernel = np.ones((5,5), np.float32)/20
    kernel2 = np.ones((10,10), np.float32)/500
    dst = cv2.filter2D(img, -1, kernel)
    dst2 = cv2.filter2D(img, -1, kernel2)

    cv2.imshow("img", dst2)
    cv2.waitKey(0)

    plt.subplot(121),plt.imshow(cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)),plt.title('Dst2')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()

img = cv2.imread('imgs/gear_nozzle_far.jpg', 1)
grayscale_to_binary(img)

#subtract_background(img)

# # Read image
# im = cv2.imread("imgs/gear_side.jpg")

# small = cv2.resize(im, (0,0), fx=0.2, fy=0.2) 
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200


# # Filter by Area.
# params.filterByArea = True
# params.minArea = 5

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.9

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# # Create a detector with the parameters
# detector = cv2.SimpleBlobDetector_create(params)


# # Detect blobs.
# keypoints = detector.detect(small)
# print(len(keypoints))
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# # the size of the circle corresponds to the size of blob

# im_with_keypoints = cv2.drawKeypoints(small, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Show blobs
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)