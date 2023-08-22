from interval import Interval
import numpy as np


class IntervalMatrix:
    """
    Matrix, each element of which is an interval
    Methods of this class implements gershgorin bounds

    Attributes:
        values: array of intervals
        shape: shape of the data
    """
    def __init__(self, values=None, shape=None):
        """
        Parameters:
            values: np.array or None
                array of intervals
                if None, array of provided shape with [-inf, inf] as values is created
            shape: tuple or None
                shape of the array
                This parameter is ignored if values is not None
                if None and values is None, (1, 1) is used
        """
        if values is not None:
            self.values = values
            self.shape = self.values.shape

            if len(self.shape) != 2:
                raise ValueError(f'Incorrect shape, got {self.shape}')
        else:
            if shape is None:
                shape = (1, 1)
            self.values = np.empty(shape, dtype=object)
            self.values[:, :] = Interval([float('-inf'), float('inf')])
            self.shape = shape

    @staticmethod
    def value_to_interval_matrix(value):
        """
        Converts value to IntervalMatrix
        Parameters:
            value: int, float, Interval or IntervalMatrix
                value to be converted
        Returns:
            Converted value
        """
        if isinstance(value, IntervalMatrix):
            return value
        if isinstance(value, Interval):
            return IntervalMatrix(values=np.array([[value]]))
        if isinstance(value, (int, float)):
            return IntervalMatrix(values=np.array([[Interval.valueToInterval(value)]]))

    @property
    def T(self):
        return IntervalMatrix(values=self.values.T)

    def is_scalar(self):
        """
        Returns:
             True if matrix has shape (1, 1)
             False otherwise
        """
        return self.shape == (1, 1)

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
        return IntervalMatrix(values=-self.values)

    def __add__(self, other):
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=self.values + other.values)

    def __radd__(self, other):
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=self.values + other.values)

    def __sub__(self, other):
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=self.values - other.values)

    def __rsub__(self, other):
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=other.values - self.values)

    def __mul__(self, other):
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=self.values * other.values)

    def __rmul__(self, other):
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=self.values * other.values)

    def __pow__(self, power):
        power = IntervalMatrix.value_to_interval_matrix(power)
        if power.shape != (1, 1):
            raise ValueError(f'Only scalar powers are supported, got shape = {power.shape}')
        return IntervalMatrix(values=self.values ** power.values[0, 0])

    @staticmethod
    def sin(interval_matrix):
        vec_func = np.vectorize(Interval.sin)
        return IntervalMatrix(values=vec_func(interval_matrix.values))

    @staticmethod
    def cos(interval_matrix):
        vec_func = np.vectorize(Interval.cos)
        return IntervalMatrix(values=vec_func(interval_matrix.values))

    @staticmethod
    def exp(interval_matrix):
        vec_func = np.vectorize(Interval.exp)
        return IntervalMatrix(values=vec_func(interval_matrix.values))

    @staticmethod
    def ln(interval_matrix):
        vec_func = np.vectorize(Interval.ln)
        return IntervalMatrix(values=vec_func(interval_matrix.values))

    @staticmethod
    def log(interval_matrix):
        return IntervalMatrix.ln(interval_matrix)

    def dot(self, other):
        if self.is_scalar() or other.is_scalar():
            return self * other
        other = IntervalMatrix.value_to_interval_matrix(other)
        return IntervalMatrix(values=self.values.dot(other.values))

    def sign(self):
        """
        Calculates lower and upper bounds using gershgorin theorem and
        gets corresponding convex information

        Returns:
            True if object is psd
            False if object is nsd
            None if none of the above
        """
        return self.is_gershgorin_convex()

    def is_gershgorin_convex(self):
        """
        Calculates lower and upper bounds using gershgorin theorem and
        gets corresponding convex information

        Returns:
            True if object is psd
            False if object is nsd
            None if none of the above
        """
        gershgorin_lower_bound = self._get_gershgorin_lower_bound()
        if gershgorin_lower_bound >= 0:
            return True
        gershgorin_upper_bound = self._get_gershgorin_upper_bound()
        if gershgorin_upper_bound <= 0:
            return False
        return None

    def _get_gershgorin_lower_bound(self):
        """
        Returns:
            Lower bound for eigenvalues of matrix, calculated by Gershgorin theorem
        """
        abs_matrix = np.abs(self.values)
        np.fill_diagonal(abs_matrix, 0)
        diagonal = self.values.diagonal()
        interval_of_bounds = diagonal - abs_matrix.sum(axis=0)
        return np.min([interval[0] for interval in interval_of_bounds])

    def _get_gershgorin_upper_bound(self):
        """
        Returns:
             Upper bound for eigenvalues of matrix, calculated by Gershgorin theorem
        """
        abs_matrix = np.abs(self.values)
        np.fill_diagonal(abs_matrix, 0)
        interval_of_bounds = self.values.diagonal() + abs_matrix.sum(axis=0)
        return np.max([interval[1] for interval in interval_of_bounds])


    @staticmethod
    def matrix_power(matrix, power):
        power = IntervalMatrix.value_to_interval_matrix(power)
        if power.shape != (1, 1):
            raise ValueError(f'Only scalar powers are supported, got shape = {power.shape}')
        power = power.values[0, 0]
        if power[0] != power[1]:
            raise ValueError(f'Only scalar powers are supported, got interval: {power}')
        power = power[0]
        return IntervalMatrix(values=np.linalg.matrix_power(matrix.values, power))

    @staticmethod
    def eye(n):
        """
        Returns:
            Identity matrix of size n x n
        """
        return IntervalMatrix(values=np.diag(np.full(shape=(n,), fill_value=Interval([1, 1]), dtype=object)))
