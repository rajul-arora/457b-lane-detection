from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

double_line_numbers = []


def draw_line_in_matrix(matrix, coordinate_pairs, is_double_line):
    for coordinate_pair in coordinate_pairs:
        matrix[coordinate_pair[0], coordinate_pair[1]] = 1
        if is_double_line:
            matrix[coordinate_pair[0] - 1, coordinate_pair[1] - 1] = 1


def generate_output_matrix_file_for_input_file(file_name):
    half_coordinate_pair_lists = []
    with open(file_name) as input_file:
        for line in input_file:
            half_coordinate_pair_list = [elt.strip() for elt in line.split(' ')]
            for i in range(0, len(half_coordinate_pair_list)):
                if half_coordinate_pair_list[i] and half_coordinate_pair_list[i] != 'dy':
                    half_coordinate_pair_list[i] = int(half_coordinate_pair_list[i])
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

        min_x = min(x[0], x[-1])
        max_x = max(x[0], x[-1])
        num_x = max_x - min_x
        lane_pixel_x_coordinates = np.linspace(min_x, max_x, num=num_x)

        f = interp1d(x, y)
        lane_pixel_y_coordinates = f(lane_pixel_x_coordinates)

        assert len(lane_pixel_x_coordinates) == len(lane_pixel_y_coordinates)
        lane_pixel_coordinate_pairs = []
        for j in range(0, len(lane_pixel_x_coordinates)):
            pair = (int(round(lane_pixel_x_coordinates[j], 0)), int(round(lane_pixel_y_coordinates[j], 0)))
            lane_pixel_coordinate_pairs.append(pair)

        lines.append(lane_pixel_coordinate_pairs)
        line_index += 1

    for line in lines:
        print(line)
        print('\n')

    np.set_printoptions(linewidth=3000)
    matrix = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int)
    line_index = 0
    for line in lines:
        draw_line_in_matrix(matrix, line, line_index in double_line_numbers)
        line_index += 1

    plt.plot([x[0] for x in lines[0]], [x[1] for x in lines[0]], [x[0] for x in lines[1]], [x[1] for x in lines[1]],
             [x[0] for x in lines[2]], [x[1] for x in lines[2]])
    plt.axis([0, IMAGE_WIDTH, 0, IMAGE_HEIGHT])
    plt.gca().invert_yaxis()
    plt.show()
    # f = open('f00000matrix.txt', 'w')
    # for row in matrix:
    #     f.write(str(row) + '\n')
    # f.close()

os.chdir("lane_images/cordova1_input_coordinates/")
for file in glob.glob("*.txt"):
    generate_output_matrix_file_for_input_file(file)

print('done!')
