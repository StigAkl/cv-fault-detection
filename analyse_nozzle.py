import cv2
import numpy as np;
from matplotlib import pyplot as plt

def process_image(img):
        binary = grayscale_to_binary(img)
        mblur, mblur2 = median_blur(binary)
        cv2.imwrite('binary.jpg', mblur2)
        contours(binary, img)
        #plot(1, mblur, mblur2)
        print("processing")
        
def contours(img, original):

    test_img = cv2.imread('binary.jpg')
    empty_image = np.zeros(test_img.shape)
    
    imgray = cv2.cvtColor(test_img,cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(imgray,130,255,0)

    _,contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(empty_image, contours, -1, (0,255,0),1)

    plot(1, empty_image, test_img)

def grayscale_to_binary(img):
    #Resize image
    resized_image = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    blur = cv2.GaussianBlur(resized_image,(5,5),0)
    
    #region_of_interest
    roi = resized_image[200:550,:]

    #Threshold
    #Resource: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    ret3,th3 = cv2.threshold(blur,105,256,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
    ret, binary_thresh = cv2.threshold(roi,110,256,cv2.THRESH_BINARY)
    return binary_thresh

def canny_edge(img):
    return cv2.Canny(img, 0,0)

def median_blur(img):
    kernel = np.ones((5,5), np.float32)/20
    kernel2 = np.ones((3,3), np.float32)/2
    dst = cv2.filter2D(img, -1, kernel)
    dst2 = cv2.filter2D(img, -1, kernel2)

    return dst, dst2

def plot(cols, *imgs):
        fig = plt.figure()
        num_images = len(imgs)
        print(num_images)
        n = 0
        for img in imgs:
            a = fig.add_subplot(cols, np.ceil(num_images/float(cols)), n+1)
            n = n+1
            plt.imshow(img, cmap='gray')
            a.set_title("Image" + str(n))
        fig.set_size_inches(np.array(fig.get_size_inches()) * num_images)
        plt.show()


if __name__ == '__main__':
    img = cv2.imread('imgs/gear_nozzle_far.jpg', cv2.IMREAD_GRAYSCALE)
    process_image(img)
