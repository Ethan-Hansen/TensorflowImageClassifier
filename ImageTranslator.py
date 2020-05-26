import cv2
import math
import numpy
import imutils
from matplotlib import pyplot as plt 
    
#reads in image
src_image = cv2.imread('C:\Users\hansn\Pictures\Mark1.png')
#4 corners of image
src_pts = numpy.float32([[335, 102], [335, 597], [834, 102], [834, 597]])
dimensions = src_image.shape

def rotateImage(alpha, beta, gamma, dx, dy, dz, f, w, h):
    
    alpha = (alpha - 90) * math.pi/180
    beta = (beta - 90) * math.pi/180
    gamma = (gamma - 90) * math.pi/180
    
    A1 = numpy.array([[1, 0, -w/2],
    [0, 1, -h/2],
    [0, 0, 0],
    [0, 0, 1]])

    RX =  numpy.array([[1, 0, 0, 0],
    [0, math.cos(alpha), -math.sin(alpha), 0],
    [0, math.sin(alpha), math.cos(alpha), 0],
    [0, 0, 0, 1]])
    
    RY =   numpy.array([[math.cos(beta), 0, -math.sin(beta), 0],
           [0, 1, 0, 0],
           [math.sin(beta), 0, math.cos(beta), 0],
           [0, 0, 0, 1]])
    
    RZ = numpy.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
           [math.sin(gamma), math.cos(gamma), 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])
    
    R = numpy.dot(numpy.dot(RX, RY), RZ)
    
    T = numpy.array([[1, 0, 0, dx],
         [0, 1, 0, dy],
         [0, 0, 1, dz],
         [0, 0, 0, 1]])
    
    A2 = numpy.array([[f, 0, w/2, 0],
          [0, f, h/2, 0],
          [0, 0, 1, 0]])
    
    return numpy.dot(A2, numpy.dot(T, numpy.dot(R, A1)))
    
x = 1
for z in range(300, 1000, 100):    
    for i in range(100, 500, 50):
        for j in range(100, 500, 50):
            for k in range(100, 500, 10):
                trans = rotateImage(i, j, k, 0, 0, z, 200, dimensions[1], dimensions[0])
                result = cv2.warpPerspective(src_image, trans, (dimensions[1], dimensions[0]))
                plt.imshow(result)
                plt.show()
                
                filename = "D:\School\Senior Design\TestImages\Is_Image\%s.png" % (x)
                cv2.imwrite(filename, result)
                x = x + 1