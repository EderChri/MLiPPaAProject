import random

import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from tqdm import tqdm
import constants
from utils import load_variable_len_data
from skspatial.objects import Line, Sphere


def circle_line_intersection(circle_center, circle_radius, line_point1, line_point2):
    """
    Calculates the intersection between a segment of a line and a circle, used for 2D data only
    :param circle_center: Tuple, center of the circle
    :param circle_radius: Float, radius of the circle
    :param line_point1: Tuple, first point the line segment goes through
    :param line_point2: Tuple, second point the line segment goes through
    :return: Tuple of x and y coordinate of the intersection
    """
    circle = Point(circle_center).buffer(circle_radius).boundary
    line = LineString([line_point1, line_point2])
    intersection = circle.intersection(line)
    return intersection.x, intersection.y


def sphere_line_intersection(sphere_center, sphere_radius, line_point1, line_point2):
    """
    Calculates the intersection between a line and a sphere, only used for 3D data
    :param sphere_center: Tuple, center of the sphere
    :param sphere_radius: Float, radius of the sphere
    :param line_point1: Tuple, first point of the line
    :param line_point2: Tuple, second point of the line
    :return: Tuple of Tuples. Each Tuple is an intersection of the line with the sphere
    """
    sphere = Sphere(sphere_center, sphere_radius)
    line = Line(line_point1, line_point2)
    intersection1, intersection2 = sphere.intersect_line(line)
    return intersection1, intersection2


def angle_to_vector(angle_in_radians):
    """
    Converts an angle to a unit vector, compatible with 2D and 3D data
    :param angle_in_radians: Float
    :return: The vector as an array, containing the coordinates
    """
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


def print_data(dict, output=f"output_{constants.DIMENSION}d.txt", parameter=False):
    """
    Auxilliary function to print the generated data
    :param dict: Dictionary of data to print
    :param output: str, name of the file to write data to
    :param parameter: Boolean, flag indicating whether the labels file is written or not
    :return:
    """
    with open(output, "w") as file:
        for key, value in dict.items():
            if not parameter:
                value = value[1::]
            value = ','.join(map(str, value))
            file.write(f"{key},{value}\n")
    print("Successfully written data")


def add_noise(point, noise):
    """
    Adds noise drawn from gaussian to a data point
    :param point: Float, one coordinate of a point
    :param noise: Float, standard deviation of gaussian
    :return: Float, point with added noise
    """
    return point + np.random.normal(0, noise, 1)[0]


def generate_data(nr_events=50_000, min_nr_tracks=2, max_nr_tracks=3, noise=0.1):
    """
    Function to generate data of trajectories of secondary particles in particle collision. Works for 2D and 3D data
    :param nr_events: int, number of events to generate
    :param min_nr_tracks: int, minimal number of tracks per event
    :param max_nr_tracks: int, maximal number of tracks per event
    :param noise: float, standard deviation of the noise to add to the event.
    :return: Tuple of dictionaries, containing the data and the labels for the data
    """
    if constants.DIMENSION == 2:
        origin = (0, 0)
    if constants.DIMENSION == 3:
        origin = [0, 0, 0]
    detector_rad = constants.DETECTOR_RADII
    # Create list to fill with values of circle/sphere-line intersections
    detector_intersect = [0 for _ in range(max_nr_tracks * len(detector_rad))]
    event_dict = {}
    parameter_dict = {}
    for event in tqdm(range(nr_events)):
        event_dict[event] = [event]
        parameter_dict[event] = []
        # Only have variable number of events if min_tracks not equal max_tracks
        nr_tracks = max_nr_tracks if min_nr_tracks == max_nr_tracks else np.random.randint(min_nr_tracks, max_nr_tracks)
        for track in range(nr_tracks):
            random_index = 0
            # For 3D data two intersections are generated for each detector-track intersection.
            # Randomly choose one of the two, but always choose the same for the same track
            if .5 < random.random():
                random_index = 1
            if constants.DIMENSION == 2:
                track_angle = random.uniform(-np.pi, np.pi)
                track_angle_print = track_angle
            if constants.DIMENSION == 3:
                track_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
                track_angle_print = f"{track_angle[0]};{track_angle[1]}"
            parameter_dict[event].append(track)
            parameter_dict[event].append(track_angle_print)
            # Scale the unit vector, so it intersects for sure with all detectors
            track_vector = constants.DETECTOR_RADII[4] * angle_to_vector(track_angle)
            for detector, rad in enumerate(detector_rad):
                if constants.DIMENSION == 2:
                    detector_intersect[track] = circle_line_intersection(origin, rad, origin, track_vector)
                if constants.DIMENSION == 3:
                    detector_intersect[track] = sphere_line_intersection(origin, rad, origin, track_vector)[
                        random_index]
                event_dict[event].append(add_noise(detector_intersect[track][0], noise))
                event_dict[event].append(add_noise(detector_intersect[track][1], noise))
                if constants.DIMENSION == 3:
                    event_dict[event].append(add_noise(detector_intersect[track][2], noise))
                event_dict[event].append(track)
    return event_dict, parameter_dict


def plot_circle(center, radius, color='blue', alpha=0.5, ax=None):
    """
    Auxiliary function to plot a circle
    :param center: Tuple, center of the circle
    :param radius: Float, radius of the circle
    :param color: str, color or the circle
    :param alpha: Float, alpha value of the circle
    :param ax: ax to plot the circle on
    :return:
    """
    circle = plt.Circle(center, radius, color=color, alpha=alpha, fill=False)
    ax.add_artist(circle)


def plot_first_row(data=f"output_{constants.DIMENSION}d.txt"):
    """
    Function to visualise and save the first line of a 2D data set
    :param data: path to the csv file
    :return:
    """
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
    plt.title("Visualization of one Event")
    plt.savefig(f"data_vis_{constants.DIMENSION}d.png")
    plt.show()


def plot_histogram(path=f"parameter_{constants.DIMENSION}d.txt"):
    """
    Function to visualise and save a histogram of the track parameters. Intended as a sanity check
    :param path: path to the csv file
    :return:
    """
    data = load_variable_len_data(path)
    plot_data = data.iloc[:, 2].tolist()
    for i in range(4, (len(data.columns)) - 1, 2):
        plot_data.extend(data.iloc[:, i].tolist())
    if constants.DIMENSION == 2:
        plot_data = [float(value) for value in plot_data if value is not None]
    if constants.DIMENSION == 3:
        split_data = [value.split(';') for value in plot_data if isinstance(value, str)]
        plot_data = [float(value) for sublist in split_data for value in sublist]
    plt.hist(plot_data, bins=7)
    plt.title(f"Histogram of Track Parameter for {constants.DIMENSION}D data")
    plt.savefig(f"histogram_{constants.DIMENSION}d.png")
    plt.show()


if __name__ == '__main__':  #
    np.random.seed(42)
    data, parameter = generate_data(nr_events=constants.NR_EVENTS,
                                    max_nr_tracks=constants.MAX_NR_TRACKS,
                                    min_nr_tracks=constants.MIN_NR_TRACKS)
    print_data(data, f"output_{constants.DIMENSION}d.txt")
    print_data(parameter, f"parameter_{constants.DIMENSION}d.txt", parameter=True)
    if constants.DIMENSION == 2:
        plot_first_row(f"output_{constants.DIMENSION}d.txt")
    plot_histogram()
