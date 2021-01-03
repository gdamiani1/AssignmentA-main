from PIL import Image
import cv2 as cv
import numpy as np


def cropobject(imagearray, left_start: int):
    """
    Outputs the coordinates of the edges of an element from the initial image.
    """
    crop_left = -1
    for l in range(left_start, width):
        for h in range(height):
            if imagearray[h][l][0] <= 149:
                crop_left = l
                break
            if l == width - 1:
                return 0, 0, 0, -1
        if crop_left == l:
            break

    crop_right = 0
    b_pixel = 0
    for r in range(crop_left, width):
        i = 1
        while i < height:
            if imagearray[i][r][0] > 149:
                b_pixel = 0
            else:
                b_pixel = 1
                break
            i += 1
        if b_pixel == 0:
            crop_right = r
            break
        if r == width:
            crop_right = width

    crop_top = -1
    for t in range(0, height):
        for w in range(crop_left, crop_right):
            if imagearray[t][w][0] <= 149:
                crop_top = t
                break
            if t == height:
                return 0
        if crop_top == t:
            break

    crop_bottom = -1
    b_pixel = 0
    for b in range(crop_top, height):
        i = 1
        while i < crop_right:
            if imagearray[b][i][0] > 149:
                b_pixel = 0
            else:
                b_pixel = 1
                break
            i += 1
        if b_pixel == 0:
            crop_bottom = b
            break
        if b == height:
            crop_bottom = height

    return crop_top, crop_bottom, crop_left, crop_right


def square_paste(img, side, crop_height, crop_width):
    """
    Pastes the cropped image of an element into a square array
    """
    square = np.full((side, side, 3), 255)
    if side == crop_height:
        start = int((side - crop_width) / 2)
        for h in range(0, crop_height):
            for w in range(0, crop_width):
                if img[h][w][0] <= 149:
                    square[h][start + w] = [0, 0, 0]
                else:
                    square[h][start + w] = [255, 255, 255]
    elif side == crop_width:
        start = int((side - crop_height) / 2)
        for h in range(0, crop_height):
            for w in range(0, crop_width):
                if img[h][w][0] <= 149:
                    square[start + h][w] = [0, 0, 0]
                else:
                    square[start + h][w] = [255, 255, 255]
    return square


def crop(img_name):
    img = cv.imread(img_name)
    global height, width, dim
    height, width, dim = img.shape
    element = []

    i = 0
    while i != -1:
        croped_img = cropobject(img, i)
        if croped_img[3] == -1:
            i = -1
            continue
        else:
            i = croped_img[3]

        img_crop = img[croped_img[0]:croped_img[1], croped_img[2]:croped_img[3]]
        crop_height = croped_img[1] - croped_img[0]
        crop_width = croped_img[3] - croped_img[2]
        if crop_height >= crop_width:
            side = crop_height
        else:
            side = crop_width
        img_square = square_paste(img_crop, side, crop_height, crop_width)
        element.append(img_square)

    numbering = 0
    for e in element:
        cv.imwrite(f'{numbering}.png', e)
        resize = np.array(Image.open(f'{numbering}.png').resize((32, 32)))
        for h in range(0, 32):
            for w in range(0, 32):
                if resize[h][w][0] <= 200:
                    resize[h][w] = [0, 0, 0]
                else:
                    resize[h][w] = [255, 255, 255]
            cv.imwrite(f'{numbering}.png', resize)
        numbering += 1

    return numbering
