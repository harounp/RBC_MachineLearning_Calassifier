import code
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import datetime
import random
import time
import matplotlib.patches as patches
import scipy
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import fijibin
import fijibin.macro

def watershed(image,directory):
    fijibin.BIN
    background_pixels = np.count_nonzero(image == 0)
    foreground_pixels = np.count_nonzero(image == 255)
    macro_0 = """open("/Users/menakamel/Desktop/Celloscope-segmentation/{}/Picture_Logs/filled_holes.png");
    run("Make Binary");
    saveAs("PNG", "/Users/menakamel/Desktop/Celloscope-segmentation/{}/Picture_Logs/watersheded.png");
    """.format(directory,directory)
    macro_1 = """open("/Users/menakamel/Desktop/Celloscope-segmentation/{}/Picture_Logs/filled_holes.png");
    run("Make Binary");
    run("Invert");
    run("Watershed");
    saveAs("PNG", "/Users/menakamel/Desktop/Celloscope-segmentation/{}/Picture_Logs/watersheded.png");
    """.format(directory,directory)
    macro_2 = """open("/Users/menakamel/Desktop/Celloscope-segmentation/{}/Picture_Logs/filled_holes.png");
    run("Make Binary");
    run("Watershed");
    saveAs("PNG", "/Users/menakamel/Desktop/Celloscope-segmentation/{}/Picture_Logs/watersheded.png");
    """.format(directory,directory)
    # import code; code.interact(local=dict(globals(), **locals()))
    if get_background_to_forground_ratio(image) > 60:
        macro = macro_0
    elif background_pixels>foreground_pixels:
        macro = macro_1
    else:
        macro = macro_2
    fijibin.macro.run(macro)

def get_background_to_forground_ratio(image_binary):
    background_size = np.count_nonzero(image_binary == 255)
    foreground_size = np.count_nonzero(image_binary == 0)
    backgroud_ratio = (background_size/float(background_size+foreground_size))*100
    return backgroud_ratio

def apply_watershed_mask(picture,directory):
    watershed_mask = cv2.imread((directory+"/Picture_Logs/watersheded.png"), cv2.IMREAD_GRAYSCALE)
    watershed_mask = np.where(watershed_mask==255, 1, watershed_mask)
    # import code; code.interact(local=dict(globals(), **locals()))
    watershed_mask[:] = [abs(x - 1) for x in watershed_mask]
    watershed_mask = np.where(watershed_mask==255, 1, watershed_mask)

    watersheded_grey_img = np.multiply(watershed_mask,np.array(picture))
    watersheded_grey_img = np.where(watersheded_grey_img==0, 255, watersheded_grey_img)
    cv2.imwrite((directory+"/Picture_Logs/watersheded_grey.png"),watersheded_grey_img)
    return watersheded_grey_img

def get_number_of_children(hierarchy):
    num_children = np.zeros(len(hierarchy[0]))
    i = 0
    for h in hierarchy[0]:
        parent = h[3]
        num_children[parent] +=1
        i+=1
    return num_children

def get_contour_coordinates(contours):
    contour_coordinates = []
    for contour in contours:
        x_coords = []
        y_coords = []
        for pixel in contour:
            x_coords.append(pixel[0][0])
            y_coords.append(pixel[0][1])
        contour_coordinates.append([x_coords,y_coords])
    return contour_coordinates

def make_new_directory():
    currentDT = datetime.datetime.now()
    time_stamp = str(currentDT).replace(':', '_').replace('.', '_')
    date = time_stamp.split(' ')[0]
    time = '-'.join(time_stamp.split(' ')[1].split('_')[0:3])
    directory = "New_Segmented_Images/" + date+'_'+time
    os.makedirs(directory)
    os.makedirs(directory + "/Picture_Logs")
    return directory

def draw_boundaries(boundaries, image_negative,directory):
    boundary_image = image_negative.copy()
    for boundary in boundaries:
        x_min = boundary[0]
        y_min = boundary[1]
        x_max = boundary[2]
        y_max = boundary[3]
        cv2.rectangle(boundary_image, (x_min, y_min), (x_max, y_max), (0,0,153), 3)
    plt.imshow(boundary_image)
    plt.show(block=False)
    cv2.imwrite((directory+'/Picture_Logs/boundaries.png'),boundary_image)

def crop_cells(image_gray,cell_ids,contours_ws,directory):
    boundaries = []
    contour_number = 1
    contour_coordinates = get_contour_coordinates(contours_ws)
    for contour in contours_ws:
        if contour_number == 1 :
            contour_number += 1
            continue
        temp = cell_ids.copy()
        temp = np.where(temp!=contour_number, 0, temp)
        temp = np.where(temp!=0, 1, temp)
        pic_name = directory+"/image_{}.png".format(contour_number-1)
        watersheded_grey_img = np.multiply(temp,np.array(image_gray))
        watersheded_grey_img = np.where(watersheded_grey_img==0, 255, watersheded_grey_img)
        x_coords,y_coords = contour_coordinates[contour_number-1]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        boundaries.append([x_min,y_min,x_max,y_max])
        try:
            watersheded_grey_img = watersheded_grey_img[y_min:y_max,x_min:x_max]
            cv2.imwrite(pic_name,watersheded_grey_img)
        except:
            continue
        contour_number += 1
    return boundaries

def add_id_to_cell(image_gray_ws,image_negative,directory):
    watershed_mask = cv2.imread((directory+"/Picture_Logs/watersheded.png"), cv2.IMREAD_GRAYSCALE)
    ret,watershed_mask = cv2.threshold(watershed_mask,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    im3, contours_ws, hierarchy_ws = cv2.findContours(watershed_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    num_children_ws = get_number_of_children(hierarchy_ws)
    contour_number_ws = 0
    cell_ids = image_negative.copy()
    for j in num_children_ws:
        cv2.drawContours(cell_ids,[contours_ws[contour_number_ws]],0,contour_number_ws+1,-1)
        contour_number_ws +=1
    id_counts = []
    for i in range(1,contour_number_ws):
        id_counts.append(np.count_nonzero(cell_ids == i))
    return [cell_ids,id_counts,contours_ws]

def remove_image_noise(image):
    ret,image_binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    background_size = np.count_nonzero(image_binary == 255)
    foreground_size = np.count_nonzero(image_binary == 0)
    backgroud_ratio = (background_size/float(background_size+foreground_size))*100
    if backgroud_ratio > 60:
        image = scipy.signal.medfilt2d(image, kernel_size=15)
    else:
        # image = scipy.signal.medfilt2d(image, kernel_size=3)
        image = image
    return image

def draw_external_boundary(binary_image):
    row_length = len(binary_image)
    column_length = len(binary_image[0])
    for i in range(column_length):
        binary_image[0][i] = 255
        binary_image[row_length-1][i] = 255
    for j in range(row_length):
        binary_image[j][0] = 255
        binary_image[j][column_length-1] = 255
    # code.interact(local=dict(globals(), **locals()))
    return binary_image

def power_law_transform(gamma, C, picture):
    transfer_function = np.zeros(256)
    r = range(256)
    correction_factor = 255.0/ (C*(255**gamma))
    for i in range(256):
        transfer_function[i] = C*(r[i]**gamma)
    correction_factor = 255.0/ (max(transfer_function))
    for i in range(256):
        transfer_function[i] = round(correction_factor*transfer_function[i])
    row_length = len(picture)
    column_length = len(picture[0])
    for j in range(row_length):
        for i in range(column_length):
            picture[j][i] = transfer_function[picture[j][i]]
    return picture


def main():
    directory = make_new_directory()
    image_name = 'RBC samples/IMG_2371.JPG'
    # image_name = 'IMG_2555.png'
    image = cv2.imread(image_name)
    image_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image_gray = remove_image_noise(image_gray)
    kernel = np.ones((10,10),np.float32)/100
    image_gray = cv2.filter2D(image_gray,-1,kernel)
    image_gray = cv2.filter2D(image_gray,-1,kernel)
    # code.interact(local=dict(globals(), **locals()))
    ret,image_binary = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    ret,image_binary_1 = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,image_binary_2 = cv2.threshold(image_gray,0,255,cv2.THRESH_OTSU)
    image_negative = cv2.bitwise_not(image_binary)
    image_negative = draw_external_boundary(image_negative)
    cv2.imwrite((directory+'/Picture_Logs/binary_image.png'),image_negative)


    im2, contours, hierarchy = cv2.findContours(image_negative,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    num_children = get_number_of_children(hierarchy)
    contour_number = 0
    start_id = 300
    cell_id = start_id
    contours_displayed = []
    temp_image = image_negative.copy()
    temp_image = temp_image.astype('int16')
    for j in num_children:
        if j <= 2 :
            cv2.drawContours(temp_image,[contours[contour_number]],0,cell_id,-1)
            contours_displayed.append(hierarchy[0][contour_number])
        contour_number +=1
        cell_id +=1
    id_counts = []
    for i in range(0,contour_number):
        id_counts.append(np.count_nonzero(temp_image == i+start_id))
    thresh = 1* np.mean(id_counts)
    # code.interact(local=dict(globals(), **locals()))
    contour_number = 0
    for j in num_children:
        if j<=2  and id_counts[contour_number] < thresh:
            cv2.drawContours(image_negative,[contours[contour_number]],0,255,-1)
            contours_displayed.append(hierarchy[0][contour_number])
        contour_number +=1
    image_reverted = cv2.bitwise_not(image_negative)
    pic_name = "filled_holes.png"
    cv2.imwrite((directory+'/Picture_Logs/'+ pic_name),image_reverted)
    watershed(image_reverted,directory)
    image_gray_ws = apply_watershed_mask(image_gray,directory)
    cell_ids, id_count, contours_ws = add_id_to_cell(image_gray_ws,image_negative,directory)
    boundaries = crop_cells(image_gray,cell_ids,contours_ws,directory)
    draw_boundaries(boundaries, image,directory)
main()
