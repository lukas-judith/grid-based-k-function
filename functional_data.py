import os
import random
import numpy as np
import matplotlib.pyplot as plt


class Function:
    """
    Discretized, one-dimensional functions with specified x, y values and type
    (e.g. what dataset the function belongs to)
    """

    def __init__(self, x_values, y_values, type_):
        self.x_values = x_values
        self.y_values = y_values
        self.type = type_
        self.neighbors = []

    def get_neighbor_list(self, all_functions):
        """
        Updates list of neighbor functions, ordered starting with functions
        of lowest distance to this function.
        """
        neighbor_dist_list = []
        # make sure this function is not included in the neigbor list
        other_functions = self.remove_function_from_list(self, all_functions)

        for f in other_functions:
            dist = self.approx_distance_L2(self, f)
            neighbor_dist_list.append([f, dist])

        # list with neighbor functions and their distance to this function (self)
        neighbor_dist_list.sort(key=self.get_dist)

        # extract the functions from the sorted list
        neighbor_list = [x[0] for x in neighbor_dist_list]
        self.neighbors = neighbor_list

    def function_equal(self, other_function):
        """
        Compares for equality with another function.
        """
        x_equal = np.array_equal(self.x_values, other_function.x_values)
        y_equal = np.array_equal(self.y_values, other_function.y_values)
        type_equal = self.type == other_function.type

        return x_equal and y_equal and type_equal

    def in_list(self, list_):
        """
        Checks if function is in given list.
        """
        for f in list_:
            if self.function_equal(f):
                return True
        return False

    @classmethod
    def get_dist(self, function_dist_pair):
        """
        Returns distance from a function-distance pair.
        Used as key for sorting a list.
        """
        func, dist = function_dist_pair
        return dist

    @classmethod
    def remove_function_from_list(self, function, list_):
        """
        Returns new list that contains all items of list_, except for function.
        """
        new_list = []

        for f in list_:
            if not f is function:
                new_list.append(f)

        return new_list

    @classmethod
    def show_functions(self, functions, folder=".", save=False):

        plt.figure(figsize=(12, 9))
        colors = ["blue", "red", "green", "orange", "grey", "purple", "yellow", "black"]

        # list containing every possible type of the functions once
        types = list(np.unique([f.type for f in functions]))

        for f in functions:
            type_idx = types.index(f.type)
            color = colors[type_idx]
            plt.plot(f.x_values, f.y_values, color=color)

        for i in range(len(types)):
            plt.plot([], label=types[i], color=colors[i])
        plt.legend()
        # either save to file or show directly
        if save:
            plt.savefig(os.path.join(folder, "all_functions.pdf"))
            plt.close()
        else:
            plt.show()

    # maybe not as classmethod?
    @classmethod
    def approx_distance_L2(self, function1, function2):
        """
        Returns the approximated L2 distance between two 1d functions,
        discretized on the same grid.
        """

        if not np.array_equal(function1.x_values, function2.x_values):
            print("Error! Functions are not evaluated on same discrete grid!")
            return

        x_values = function1.x_values

        delta_x = np.diff(x_values)
        delta_y = (function1.y_values - function2.y_values)[1:]

        dist = np.sum(delta_x * delta_y**2)
        return dist

    @classmethod
    def get_nns_all_functions(self, functions):
        """
        For all given functions, find the neighbor list,
        ordered by smallest distance.
        """
        for f in functions:
            f.get_neighbor_list(functions)


def data_to_function_objects(data):
    """
    Converts K function data into list of Function object.
    """
    functions = []

    range_of_t = data[0][0]
    type_ = data[0][3]
    # K_diff = [d[1] for d in data]
    K_diff = [d[2] for d in data]

    for i in range(len(K_diff)):
        x_values = range_of_t
        y_values = K_diff[i]
        f = Function(x_values, y_values, type_)
        functions.append(f)

    return functions
