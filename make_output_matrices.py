from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

double_line_numbers = []
value = 1


def widen(y, x, start, width, matrix):
    for i in range(width):
        matrix[y][x + start + i] = value
        matrix[y][x - start - i] = value


def widen_according_to_y_axis(y, x, is_double):
    start = 1
    if is_double:
        start = 2
    if y > 345:
        width = 3
        widen(y, x, start, width, matrix)
        widen(y-1, x, start, width, matrix)
        if is_double:
            widen(y, x, start, width + 5, matrix)
            widen(y-1, x, start, width + 5, matrix)
    if y > 330 and y <= 345:
        width = 2
        widen(y, x, start, width, matrix)
        widen(y-1, x, start, width, matrix)
        if is_double:
            widen(y, x, start, width + 5, matrix)
            widen(y-1, x, start, width + 5, matrix)
    elif y <= 330 and y > 290:
        width = 1
        widen(y, x, start, width, matrix)
        widen(y-1, x, start, width, matrix)

        if is_double:
            widen(y, x, start, width + 4, matrix)
            widen(y - 1, x, start, width + 4, matrix)
    elif y <= 290 and y > 250:
        matrix[y][x + 1] = value
        matrix[y][x - 1] = value

        matrix[y-1][x + 1] = value
        matrix[y-1][x - 1] = value

        if is_double:
            widen(y, x, start, 3, matrix)
            widen(y - 1, x, start, 3, matrix)
    elif y <= 250 and y > 225 and is_double:
        widen(y, x, start, 2, matrix)
        widen(y - 1, x, start, 2, matrix)


def draw_line_in_matrix(coordinate_pairs, is_double_line):
    for coordinate_pair in coordinate_pairs:
        if coordinate_pair[1] < 200:
            coordinate_pair = (coordinate_pair[0] - 2, coordinate_pair[1])
        if coordinate_pair[1] < 240 and coordinate_pair[1] > 200:
            coordinate_pair = (coordinate_pair[0] - 1, coordinate_pair[1])
        matrix[coordinate_pair[1], coordinate_pair[0]] = value
        matrix[coordinate_pair[1] - 1, coordinate_pair[0]] = value
        if is_double_line:
            matrix[coordinate_pair[1] - 1, coordinate_pair[0] - 1] = value
            matrix[coordinate_pair[1] - 1, coordinate_pair[0]] = value
            matrix[coordinate_pair[1] - 1, coordinate_pair[0] + 1] = value
            matrix[coordinate_pair[1] + 1, coordinate_pair[0] - 1] = value
            matrix[coordinate_pair[1] + 1, coordinate_pair[0]] = value
            matrix[coordinate_pair[1] + 1, coordinate_pair[0] + 1] = value

            matrix[coordinate_pair[1], coordinate_pair[0] - 1] = value
            matrix[coordinate_pair[1], coordinate_pair[0] + 1] = value

            matrix[coordinate_pair[1] - 1, coordinate_pair[0] - 1] = value
            matrix[coordinate_pair[1] - 1, coordinate_pair[0] + 1] = value

        widen_according_to_y_axis(coordinate_pair[1], coordinate_pair[0], is_double_line)


def generate_output_matrix_file_for_input_file(file_name, matrix):
    half_coordinate_pair_lists = []
    with open(file_name) as file:
        for line in file:
            half_coordinate_pair_list = [elt.strip() for elt in line.split(' ')]
            for i in range(0, len(half_coordinate_pair_list)):
                if half_coordinate_pair_list[i] and half_coordinate_pair_list[i] != 'dy':
                    half_coordinate_pair_list[i] = int(half_coordinate_pair_list[i])
            if type(half_coordinate_pair_list[-1]) is not int:
                half_coordinate_pair_list.pop(-1)
            half_coordinate_pair_lists.append(half_coordinate_pair_list)

    assert len(half_coordinate_pair_lists) % 2 == 0
    num_lines = len(half_coordinate_pair_lists) / 2
    line_index = 0
    lines = []
    for i in range(0, int(num_lines)):
        x = half_coordinate_pair_lists[line_index * 2]
        y = half_coordinate_pair_lists[line_index * 2 + 1]

        if x[0] == 'dy':
            double_line_numbers.append(i)
            x.pop(0)

        lane_pixel_coordinate_pairs = []
        first_pair = (int(round(x[0])), int(round(y[0])))
        last_pair = (int(round(x[-1])), int(round(y[-1])))
        lane_pixel_coordinate_pairs.append(first_pair)
        min_x = min(x[0], x[-1])
        max_x = max(x[0], x[-1])
        num_x = max_x - min_x + 1
        lane_pixel_x_coordinates = np.linspace(min_x, max_x, num=num_x)

        f = interp1d(x, y)
        lane_pixel_y_coordinates = f(lane_pixel_x_coordinates)

        assert len(lane_pixel_x_coordinates) == len(lane_pixel_y_coordinates)
        for j in range(1, len(lane_pixel_x_coordinates)-1):
            pair = (int(round(lane_pixel_x_coordinates[j], 0)), int(round(lane_pixel_y_coordinates[j], 0)))
            lane_pixel_coordinate_pairs.append(pair)
        lane_pixel_coordinate_pairs.append(last_pair)

        lines.append(lane_pixel_coordinate_pairs)
        line_index += 1

    for line in lines:
        print(line)
        print('\n')

    np.set_printoptions(linewidth=1500)
    #np.set_printoptions(suppress=True)

    line_index = 0
    for line in lines:
        for coordinate_pair in line:
            draw_line_in_matrix(line, line_index in double_line_numbers)
        line_index += 1
    #cv2.medianBlur(matrix, 5)
    #cv2.imwrite('test.jpg', matrix)
    filename = '../cordova2_output_matrices/' + file_name.split('.')[0] + 'matrix.txt'
    matrix.tofile(filename, sep=" ")


os.chdir("lane_images/cordova2_input_coordinates/")
for file in glob.glob("*.txt"):
    matrix = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int)
    generate_output_matrix_file_for_input_file(file, matrix)

print('done!')
