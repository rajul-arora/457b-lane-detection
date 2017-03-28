from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

lane_pixel_coordinate_pairs = []

half_coordinate_pair_lists = []
with open('lane_images/cordova1_lane_coordinates/f00000.txt') as file:
    for line in file:
        half_coordinate_pair_list = [elt.strip() for elt in line.split(' ')]
        for i in range(0, len(half_coordinate_pair_list)):
            if half_coordinate_pair_list[i]:
                half_coordinate_pair_list[i] = int(half_coordinate_pair_list[i])
        half_coordinate_pair_lists.append(half_coordinate_pair_list)

assert len(half_coordinate_pair_lists) % 2 == 0
num_lines = len(half_coordinate_pair_lists) / 2
line_index = 0
for i in range(0, int(num_lines)):
    x = half_coordinate_pair_lists[line_index * 2]
    y = half_coordinate_pair_lists[line_index * 2 + 1]

    min_x = min(x[0], x[-1])
    max_x = max(x[0], x[-1])
    num_x = max_x - min_x
    lane_pixel_x_coordinates = np.linspace(min_x, max_x, num=num_x)

    f = interp1d(x, y)
    lane_pixel_y_coordinates = f(lane_pixel_x_coordinates)

    assert len(lane_pixel_x_coordinates) == len(lane_pixel_y_coordinates)
    for j in range(0, len(lane_pixel_x_coordinates)):
        pair = (int(round(lane_pixel_x_coordinates[j], 0)), int(round(lane_pixel_y_coordinates[j], 0)))
        lane_pixel_coordinate_pairs.append(pair)

    line_index += 1

print(lane_pixel_coordinate_pairs)
np.set_printoptions(linewidth=800)
matrix = np.zeros(shape=(IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.int)
for coordinate_pair in lane_pixel_coordinate_pairs:
    matrix[coordinate_pair[0], coordinate_pair[1]] = 1

f = open('f00000matrix.txt', 'w')
for row in matrix:
    f.write(str(row) + '\n')# python will convert \n to os.linesep
f.close()

print('done!')