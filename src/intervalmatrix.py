from interval import Interval
import numpy as np

class IntervalMatrix:
    def __init__(self, shape=None, values=None, is_psd=None, psd_interval=None):
        if shape is not None:
            assert len(shape) == 2, f'Only matrices with two dimensions are supported, got shape = {shape}'
            self.shape = shape

        if values is None and shape is not None:
            self.data = np.empty(shape, dtype=object)
            self.data[:, :] = Interval([float('-inf'), float('inf')])
        elif values is not None:
            self.data = values
            self.shape = values.shape
        else:
            self.data = None
            self.shape = (0, 0)

        if is_psd is None:
            if psd_interval is None:
                if self.shape == (1, 1):
                    self.interval = self.data[0, 0]
                else:
                    self.interval = Interval([float('-inf'), float('inf')])
            else:
                self.interval = psd_interval
        else:
            assert psd_interval is None, f'Only one of is_psd and psd_interval should be passed, got both'
            if is_psd:
                self.interval = Interval([0, float('inf')])
            else:
                self.interval = Interval([float('-inf'), 0])

    @staticmethod
    def value_to_interval_matrix(value):
        if isinstance(value, IntervalMatrix):
            return value
        if isinstance(value, Interval):
            return IntervalMatrix(values=np.array([value]))
        if isinstance(value, (int, float)):
            return IntervalMatrix(values=np.array([Interval.valueToInterval(value)]))

    @property
    def T(self):
        if self.data is not None:
            data = self.data.T
        else:
            data = None
        return IntervalMatrix(values=data, psd_interval=self.interval)

    def __str__(self):
        if self.shape == (1, 1):
            return str(self.data[0, 0])
        result = ''

        result += 'IntervalMatrix:\n'
        result += '[\n'
        for i in range(self.shape[0]):
            result += '['
            for j in range(self.shape[1]):
                result += str(self.data[i, j])
                result += '\t'
            result += ']\n'
        result += ']\n'
        result += f'convex interval = {str(self.interval)}'
        return result

    def __neg__(self):
        if self.data is not None:
            self.data = -self.data
        self.interval = -self.interval

    def __add__(self, other):
        other = self.value_to_interval_matrix(other)
        if self.data is not None:
            values = self.data.__add__(other.data)
        else:
            values = None
        interval = self.interval.__add__(other.interval)
        return IntervalMatrix(values=values, psd_interval=interval)

    def __radd__(self, other):
        other = self.value_to_interval_matrix(other)
        if self.data is not None:
            values = self.data.__radd__(other.data)
        else:
            values = None
        interval = self.interval.__radd__(other.interval)
        return IntervalMatrix(values=values, psd_interval=interval)

    def __sub__(self, other):
        other = self.value_to_interval_matrix(other)
        if self.data is not None:
            values = self.data.__sub__(other.data)
        else:
            values = None
        interval = self.interval.__sub__(other.interval)
        return IntervalMatrix(values=values, psd_interval=interval)

    def __rsub__(self, other):
        other = self.value_to_interval_matrix(other)
        if self.data is not None:
            values = self.data.__rsub__(other.data)
        else:
            values = None

        interval = self.interval.__rsub__(other.interval)
        return IntervalMatrix(values=values, psd_interval=interval)

    def __mul__(self, other):
        other = self.value_to_interval_matrix(other)
        if self.data is not None:
            values = self.data.__mul__(other.data)
        else:
            values = None
        if isinstance(other, (int, float, Interval)):
            interval = self.interval.__mul__(other)
            return IntervalMatrix(values=values, psd_interval=interval)
        return IntervalMatrix(values=values, is_psd=None)

    def __rmul__(self, other):
        if self.data is not None:
            values = self.data.__rmul__(self.value_to_interval_matrix(other).data)
        else:
            values = None
        if isinstance(other, (int, float, Interval)):
            interval = self.interval.__rmul__(other)
            return IntervalMatrix(values=values, psd_interval=interval)
        return IntervalMatrix(values=values, is_psd=None)

    def __pow__(self, power):
        if self.data is not None and power[0] == power[1]:
            values = self.data.__pow__(power)
        else:
            values = None
        return IntervalMatrix(values=values, is_psd=None)

    def dot(self, other):
        if self.data is not None:
            if isinstance(other, (int, float, Interval)):
                values = self.data.dot(Interval.valueToInterval(other))
            else:
                values = self.data.dot(self.value_to_interval_matrix(other).data)
        else:
            values = None
        if isinstance(other, (int, float, Interval)):
            interval = self.interval.__mul__(other)
            return IntervalMatrix(values=values, psd_interval=interval)
        return IntervalMatrix(values=values, is_psd=None)

    def sign(self):
        return self.interval.sign()

    def is_gershgorin_convex(self):
        gershgorin_lower_bound = get_gershgorin_lower_bound(self.data)
        if gershgorin_lower_bound >= 0:
            return True
        gershgorin_upper_bound = get_gershgorin_upper_bound(self.data)
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
    if matrix.data is not None and power[0] == power[1]:
        values = np.linalg.matrix_power(matrix.data, int(power[0]))
    else:
        values = None
    return IntervalMatrix(values=values, is_psd=None)
