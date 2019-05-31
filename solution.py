#!/usr/bin/env python3
import argparse
import glob
import math
import os
import sys

import numpy
from PIL import Image
from numpy import asarray


def calculate_hamming_distance(element, comparing):
    element_array = list(element)
    comparing_array = list(comparing)
    if len(element_array) != len(comparing_array):
        raise AttributeError("These data cannot be compared. The input data has different sizes!")
    else:
        difference = 0
        for position in range(len(element_array)):
            if element_array[position] != comparing_array[position]:
                difference += 1
        return difference


# Computing the color average
def average(image):
    pixels = numpy.asarray(image)
    avg = 0
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            if i != 0 and j != 0:
                avg += pixels[i][j]
    return avg / 64


# Cutting the image, taking first 8x8 pixels
def cut_image(image):
    cropped_image = []
    cut_length = 8
    pixels = numpy.asarray(image)
    for i in range(len(pixels)):
        cropped_image_row = []
        for j in range(len(pixels[i])):
            if i + j <= 2 * (cut_length - 1) and i < cut_length and j < cut_length:
                cropped_image_row.append(pixels[i][j])
        if cropped_image_row.__len__() != 0:
            cropped_image.append(cropped_image_row)
    return Image.fromarray(numpy.array(cropped_image))


# The working directory, where our dataset locates
source = "input"


# Provides the possibility to obtain the list of files (in current situation - images) from the src

def get_image_list_from(src):
    image_list = []
    for filename in glob.glob(src + "/*"):
        image_list.append(filename)
    return image_list


"""
 Implementation of the greyscaling the image matrix
 The formula: gray = 0.2989 * r + 0.5870 * g + 0.1140 * b make it possible to greyscale the image
 It is a modern digital image greyscale format (BT.601)
"""


def greyscale(image):
    return image.convert("L")


# Classification algorithm
def classification(image_hashes, image_string_list, coefficients):
    # The list classification
    dublicate_images = []
    modification_images = []
    similar_images = []
    for outer in range(len(image_hashes)):
        outer_image_hash = asarray(image_hashes[outer])
        for inner in range(len(image_hashes)):
            if outer != inner:
                inner_image_hash = asarray(image_hashes[inner])
                difference = calculate_hamming_distance(outer_image_hash, inner_image_hash)
                if int(difference) == coefficients[0]:
                    if not (image_string_list[inner], image_string_list[outer]) in dublicate_images:
                        dublicate_images.append((image_string_list[outer], image_string_list[inner]))
                elif coefficients[0] < int(difference) <= coefficients[1]:
                    if not (image_string_list[inner], image_string_list[outer]) in modification_images:
                        modification_images.append((image_string_list[outer], image_string_list[inner]))
                elif coefficients[1] < int(difference) <= coefficients[2]:
                    if not (image_string_list[inner], image_string_list[outer]) in similar_images:
                        similar_images.append((image_string_list[outer], image_string_list[inner]))
                else:
                    continue
    print("Duplicate images:\n", dublicate_images, "\n")
    print("Modification images:\n", modification_images, "\n")
    print("Similar images: \n", similar_images, "\n")


"""
    Implementation of the phash algorithm
"""

# Constant, which saves the obligatory dimension for resizing the image to the matrix 9x8 before dct algorithm

scale_parameters_phash = 9, 8


# Implementing the 2D DCT algorithm
def calculate_dct(image):
    dct = []
    pixels = numpy.asarray(image)
    for i in range(len(pixels)):
        dct.append(calculate_vector_dct(pixels[i]))
    return numpy.asarray(dct)


# Implementing the 1D DCT algorithm
def calculate_vector_dct(vector):
    transformed = []

    for i in range(len(vector)):
        summary = 0
        for j in range(len(vector)):
            summary += vector[j] * math.cos(i * math.pi * (j + 0.5) / len(vector))
        summary *= math.sqrt(2 / len(vector))
        if i == 0:
            summary *= 1 / math.sqrt(2)
        transformed.append(summary)

    return transformed


# Hashing the image with such algo: compare each bit of the image to the average and if the bit is greater-equals
# than average, put to hash 1, else - 0

def phashing(image, avg):
    image_hash = []
    pixels = numpy.asarray(image)
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            if (i != 0 and j != 0) and pixels[i][j] >= avg:
                image_hash.append(1)
            else:
                image_hash.append(0)
    return image_hash


"""
 Working out the pHash algorithm. It consists of the next steps:
 - Resizing the images to the 32x32 scale
 - Greyscale the images, simultaneously implementing the DCT algorithm 
 - Computing the DCT
 - Compute the average value
 - Further reduce the DCT
 - Construct the hash
"""


def phash(database_src, coefficients):
    print("phash")
    global source
    source = database_src
    # Receiving the string list of the images, locating on the directory 'input_path'
    image_string_list = get_image_list_from(source)
    images_hash = []
    # Iterating through the image string list, working out each image, which locates on the directory 'input_path'
    for index in range(len(image_string_list)):
        # Receiving ONLY the image name (without the path to it) with extension and image title
        imagename_with_extension = os.path.basename(image_string_list[index])
        # Splitting up the extension and the image name of the image file
        imagename, extension = os.path.splitext(imagename_with_extension)
        image = Image.open(image_string_list[index])
        # Resizing the image to 32x32 pixel matrix
        image = image.resize(scale_parameters_phash, Image.ANTIALIAS)
        # Greyscaling the image
        image = greyscale(image)
        # Providing the DCT algorithm to the image transformation.             # Firstly, I have to make the array
        # from the image, then use the algorithm to each row.
        image = Image.fromarray(calculate_dct(image))
        # Extract the top-left 8x8 pixels.
        image = cut_image(image)
        # Getting the average value of the after-dct algorithm
        avg = average(image)
        # Hashing after dct image matrix
        image_hash = phashing(image, avg)
        # Appending the transformed image to the list, which will be classified by the 'classification' algorithm
        images_hash.append(image_hash)

    classification(images_hash, image_string_list, coefficients)


"""
    Implementation of the dhash algorithm
"""

# Constant, which saves the obligatory dimension for resizing the image to the matrix 9x8

scale_parameters_dhash = 9, 8


# Computing the column/row hash of the image after the greyscale and resizing procedures

def dhashing_row(image):
    image_row_hash = []
    image_pixels = numpy.asarray(image)
    for i in range(len(image_pixels)):
        for j in range(len(image_pixels[i])):
            if j + 1 < len(image_pixels[i]):
                if image_pixels[i][j] < image_pixels[i][j + 1]:
                    image_row_hash.append(1)
                else:
                    image_row_hash.append(0)
    return image_row_hash


def dhashing_column(image):
    image_column_hash = []
    image_pixels = numpy.asarray(image)
    for i in range(len(image_pixels[0])):
        for j in range(len(image_pixels)):
            if j + 1 < len(image_pixels):
                if image_pixels[j][i] < image_pixels[j + 1][i]:
                    image_column_hash.append(1)
                else:
                    image_column_hash.append(0)
    return image_column_hash


"""
 Working out the dHash algorithm. It consists of the next steps:
 - Resizing the images to the 9x8 scale
 - Greyscale the images
 - Make the row/column hashing
 - Classify the transformed images
"""


def dhash(database_src, coefficients):
    print("dhash")
    global source
    source = database_src
    # Receiving the string list of the images, locating on the directory 'input_path'
    image_string_list = get_image_list_from(source)
    images_hash = []
    # Iterating through the image string list, working out each image, which locates on the directory 'input_path'
    for index in range(len(image_string_list)):
        # Receiving ONLY the image name (without the path to it) with extension and image title
        imagename_with_extension = os.path.basename(image_string_list[index])
        # Splitting up the extension and the image name of the image file
        imagename, extension = os.path.splitext(imagename_with_extension)
        image = Image.open(image_string_list[index])
        # Greyscaling the image
        image = greyscale(image)
        # Resizing the image to 9x8 pixel matrix
        image = image.resize(scale_parameters_dhash, Image.ANTIALIAS)
        # Receiving the hash of each image
        image_row_hash = dhashing_row(image)
        image_column_hash = dhashing_column(image)
        image_hash = image_row_hash + image_column_hash
        # Appending the transformed image to the list, which will be classified by the 'classification' algorithm
        images_hash.append(image_hash)

    classification(images_hash, image_string_list, coefficients)


"""
    Implementation of the ahash algorithm
"""

# Constant, which saves the obligatory dimension for resizing the image to the matrix 8x8 before dct algorithm

scale_parameters = 8, 8


# Hashing the image with such algo: compare each bit of the image to the average and if the bit is greater-equals
# than average, put to hash 1, else - 0

def ahashing(image, avg):
    image_hash = []
    pixels = numpy.asarray(image)
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            if pixels[i][j] >= avg:
                image_hash.append(1)
            else:
                image_hash.append(0)
    return image_hash


"""
 Working out the aHash algorithm. It consists of the next steps:
 - Resizing the images to the 8x8 scale
 - Greyscale the images
 - Compute the bits
 - Construct the hash
"""


def ahash(database_src, coefficients):
    print("ahash")
    global source
    source = database_src
    # Receiving the string list of the images, locating on the directory 'input_path'
    image_string_list = get_image_list_from(source)
    image_hashes = []
    # Iterating through the image string list, working out each image, which locates on the directory 'input_path'
    for index in range(len(image_string_list)):
        # Receiving ONLY the image name (without the path to it) with extension and image title
        imagename_with_extension = os.path.basename(image_string_list[index])
        # Splitting up the extension and the image name of the image file
        imagename, extension = os.path.splitext(imagename_with_extension)
        image = Image.open(image_string_list[index])
        # Resizing the image to 8x8 pixel matrix
        image = image.resize(scale_parameters, Image.ANTIALIAS)
        # Greyscaling the image
        image = greyscale(image)
        # Computing the bits average
        avg = average(image)
        # Making the hash of the current image
        image_hash = ahashing(image, avg)
        # Appending the transformed image to the list, which will be classified by the 'classification' algorithm
        image_hashes.append(image_hash)

    classification(image_hashes, image_string_list, coefficients)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='The working directory, where we store the images')
    parser.add_argument('-f', '--func', type=str, default='dhash', help='The functions we use to solve the problem: ahash, dhash, phash. The default algorithm is dhash')
    parser.add_argument('-d', '--dub', type=int, default=0, help='Dublicate parameter. The optimal value is 0 for all algorithms')
    parser.add_argument('-m', '--mod', type=int, default=14, help='Modification parameter. The optimal values are: 6 - for ahash/phash and 14 - for dhash')
    parser.add_argument('-s', '--sim', type=int, default=32,help='Similar parameter. The optimal values are: 13 - for ahash/phash and 32 - for dhash')
    args = vars(parser.parse_args())
    print(args)
    if args['func'] == "ahash":
        ahash(args['src'], [args['dub'], args['mod'], args['sim']])
    elif args['func'] == "dhash":
        dhash(args['src'], [args['dub'], args['mod'], args['sim']])
    elif args['func'] == "phash":
        phash(args['src'], [args['dub'], args['mod'], args['sim']])
    else:
        parser.print_help()


main()
