import random

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from tqdm import tqdm
import constants
from utils import load_variable_len_data
from skspatial.objects import Line, Sphere


def circle_line_intersection(circle_center, circle_radius, line_point1, line_point2):
    circle = Point(circle_center).buffer(circle_radius).boundary
    line = LineString([line_point1, line_point2])
    intersection = circle.intersection(line)
    return intersection.x, intersection.y


def sphere_line_intersection(sphere_center, sphere_radius, line_point1, line_point2):
    sphere = Sphere(sphere_center, sphere_radius)
    line = Line(line_point1, line_point2)
    intersection, _ = sphere.intersect_line(line)
    return intersection


def angle_to_vector(angle_in_radians):
    if constants.DIMENSION == 2:
        vx = np.cos(angle_in_radians)
        vy = np.sin(angle_in_radians)
        return np.array([vx, vy])
    if constants.DIMENSION == 3:
        theta_rad, phi_rad = angle_in_radians
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)

        return np.array([x, y, z])


def print_data(dict, output="output.txt", parameter=False):
    with open(output, "w") as file:
        for key, value in dict.items():
            if not parameter:
                value = value[1::]
            value = ','.join(map(str, value))
            file.write(f"{key},{value}\n")
    print("Successfully written data")


def add_noise(point, noise):
    return point + np.random.normal(0, noise, 1)[0]


def generate_data(nr_events=50_000, max_nr_tracks=3, noise=0.1):
    if constants.DIMENSION == 2:
        origin = (0, 0)
    if constants.DIMENSION == 3:
        origin = [0, 0, 0]
    detector_rad = constants.DETECTOR_RADII
    detector_intersect = [0 for _ in range(max_nr_tracks * len(detector_rad))]
    event_dict = {}
    parameter_dict = {}
    for event in tqdm(range(nr_events)):
        event_dict[event] = [event]
        parameter_dict[event] = []
        for track in range(np.random.randint(2, max_nr_tracks)):
            if constants.DIMENSION == 2:
                track_angle = random.uniform(-np.pi, np.pi)
            if constants.DIMENSION == 3:
                track_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
            parameter_dict[event].append(track)
            parameter_dict[event].append(track_angle)
            track_vector = 5 * angle_to_vector(track_angle)
            for detector, rad in enumerate(detector_rad):
                if constants.DIMENSION == 2:
                    detector_intersect[track] = circle_line_intersection(origin, rad, origin, track_vector)
                if constants.DIMENSION == 3:
                    detector_intersect[track] = sphere_line_intersection(origin, rad, origin, track_vector)
                    event_dict[event].append(add_noise(detector_intersect[track][2], noise))
                event_dict[event].append(add_noise(detector_intersect[track][0], noise))
                event_dict[event].append(add_noise(detector_intersect[track][1], noise))
                event_dict[event].append(track)
    return event_dict, parameter_dict


def plot_circle(center, radius, color='blue', alpha=0.5, ax=None):
    circle = plt.Circle(center, radius, color=color, alpha=alpha, fill=False)
    ax.add_artist(circle)


def plot_first_row(data="output.txt"):
    circle_center = (0, 0)
    circle_radii = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots()
    for radius in circle_radii:
        plot_circle(circle_center, radius, ax=ax)

    with open(data) as f:
        data_row = f.readline().split(',')
    data_row = [float(i) for i in data_row]
    for i in range(1, len(data_row), 3):
        plt.plot(data_row[i], data_row[i + 1], marker="o", markerfacecolor="black", markeredgecolor="black")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', adjustable='datalim')
    plt.grid()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Circles with Different Radii")
    plt.show()


def plot_histogram(path="parameter.txt"):
    data = load_variable_len_data(path)
    plot_data = data.iloc[:, 2].tolist()
    for i in range(4, (len(data.columns)) - 1, 2):
        plot_data.extend(data.iloc[:, i].tolist())
    plot_data = [float(value) for value in plot_data if value is not None]
    plt.hist(plot_data, bins=7)
    plt.show()


if __name__ == '__main__':  #
    np.random.seed(42)
    # data, parameter = generate_2Ddata(nr_events=500, max_nr_tracks=20)
    data, parameter = generate_data(nr_events=500, max_nr_tracks=20)
    print_data(data)
    print_data(parameter, "parameter.txt", parameter=True)
    plot_first_row("output.txt")
    plot_histogram()
