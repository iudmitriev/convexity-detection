from interval import Interval
import numpy as np


class MatrixOfIntervals:
    """
        Matrix, each element of which is an interval

        Attributes:
            values: array of intervals
            shape: shape of the data
        """
    def __init__(self, values=None, shape=None):
        if values is not None:
            self.values = values
            self.shape = self.values.shape
        elif shape is not None:
            self.values = np.empty(shape, dtype=object)
            self.values[:, :] = Interval([float('-inf'), float('inf')])
            self.shape = shape
        else:
            raise ValueError(f'Both values and shape are None, failed to create MatrixOfIntervals')

    @staticmethod
    def value_to_matrix_of_intervals(value):
        if isinstance(value, MatrixOfIntervals):
            return value
        if isinstance(value, Interval):
            return MatrixOfIntervals(values=np.array([[value]]))
        if isinstance(value, (int, float)):
            return MatrixOfIntervals(values=np.array([[Interval.valueToInterval(value)]]))

    @property
    def T(self):
        return MatrixOfIntervals(values=self.values.T)

    def __str__(self):
        if self.shape == (1, 1):
            return str(self.values[0, 0])
        result = ''

        result += 'MatrixOfIntervals:\n'
        result += '[\n'
        for i in range(self.shape[0]):
            result += '['
            for j in range(self.shape[1]):
                result += str(self.values[i, j])
                result += '\t'
            result += ']\n'
        result += ']\n'
        return result

    def __neg__(self):
        return MatrixOfIntervals(values=-self.values)

    def __add__(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=self.values + other.values)

    def __radd__(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=self.values + other.values)

    def __sub__(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=self.values - other.values)

    def __rsub__(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=other.values - self.values)

    def __mul__(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=self.values * other.values)

    def __rmul__(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=self.values * other.values)

    def __pow__(self, power):
        power = MatrixOfIntervals.value_to_matrix_of_intervals(power)
        if power.shape != (1, 1):
            raise ValueError(f'Only scalar powers are supported, got shape = {power.shape}')
        return MatrixOfIntervals(values=self.values ** power.values[0, 0])

    def dot(self, other):
        other = MatrixOfIntervals.value_to_matrix_of_intervals(other)
        return MatrixOfIntervals(values=self.values.dot(other.values))

    def sign(self):
        return self.is_gershgorin_convex()

    def is_gershgorin_convex(self):
        gershgorin_lower_bound = get_gershgorin_lower_bound(self.values)
        if gershgorin_lower_bound >= 0:
            return True
        gershgorin_upper_bound = get_gershgorin_upper_bound(self.values)
        if gershgorin_upper_bound <= 0:
            return False
        return None


def get_gershgorin_lower_bound(matrix_of_intervals):
    abs_matrix = np.abs(matrix_of_intervals)
    np.fill_diagonal(abs_matrix, 0)
    diagonal = matrix_of_intervals.diagonal()
    interval_of_bounds = diagonal - abs_matrix.sum(axis=0)
    return np.min([interval[0] for interval in interval_of_bounds])


def get_gershgorin_upper_bound(matrix_of_intervals):
    abs_matrix = np.abs(matrix_of_intervals)
    np.fill_diagonal(abs_matrix, 0)
    interval_of_bounds = matrix_of_intervals.diagonal() + abs_matrix.sum(axis=0)
    return np.max([interval[1] for interval in interval_of_bounds])


def matrix_power(matrix, power):
    power = MatrixOfIntervals.value_to_matrix_of_intervals(power)
    if power.shape != (1, 1):
        raise ValueError(f'Only scalar powers are supported, got shape = {power.shape}')
    power = power.values[0, 0]
    if power[0] != power[1]:
        raise ValueError(f'Only scalar powers are supported, got interval: {power}')
    power = power[0]
    return MatrixOfIntervals(values=np.linalg.matrix_power(matrix.values, power))
