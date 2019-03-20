
#Credit to Ahmed Fajem, on of my partners in this project
#This code was used to expand the burr cell data set from 400 to 
#4000 images using image rotations and reflections


from os import listdir
from os.path import isfile, join
import cv2
import math
import numpy as np
#image = cv2.imread('cat.jpg')
#print (image)
#cv2.imshow('image' , image)
#cv2.waitKey(0)


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat



# to read all the images from a folder
mypath=r'C:\Users\Pierre H\Downloads\Burr\Burr'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)

for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]),1 ) #Reading the image from the folder path chosen
e=0
for q in images:
    q[np.where((q == [0,0,0]).all(axis = 2))] = [255,255,255]
    j=0
    while j < 360:
        rotate_image(q,j)
        cv2.imwrite(r'C:\Users\Pierre H\PycharmProjects\OptimizingCelloScopeModel\venv\data\burr\Cell%s.jpg' % e, rotate_image(q,j))
        e+=1
        cv2.imwrite(r'C:\Users\Pierre H\PycharmProjects\OptimizingCelloScopeModel\venv\data\burr\Cell%s.jpg' % e, rotate_image(cv2.flip(q,1),j))
        e+=1
        j+=90
        



#showing the images
#cv2.imshow('image 1' ,images[0])
#cv2.waitKey(0)
#x= len(images)
#print(x)


#cv2.destroyAllWindow(image) 



#cv2.imshow('image' , image)
#cv2.imshow('rotated', rotate_image(image,45 ))


#To save the image to the wanted directory
#cv2.imwrite(r'C:\Users\user\Desktop\Electrical Engineering\4th year\4BI6\Project test\rotated.jpg', rotate_image(image,45 ))

#vertical_img=image.copy()
#vertical_img=cv2.flip(image,1)
#cv2.imshow('fliped', vertical_img)
#cv2.waitKey(0)



