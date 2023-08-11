from interval import Interval
from interval_matrix import IntervalMatrix
from psd_interval_information import PsdIntervalInformation


class IntervalMatrixWithPsdInterval:
    def __init__(self, matrix=None, psd_interval=None):
        if matrix is None:
            matrix = IntervalMatrix()
        self.matrix = matrix
        if psd_interval is None:
            psd_interval = PsdIntervalInformation()
        self.interval = psd_interval

        is_gershgorin_psd = self.matrix.is_gershgorin_convex()
        if is_gershgorin_psd is not None and self.interval.sign() is None:
            self.interval = PsdIntervalInformation(is_psd=is_gershgorin_psd)

    @staticmethod
    def value_to_interval_matrix_with_psd_interval(value):
        if isinstance(value, IntervalMatrixWithPsdInterval):
            return value
        if isinstance(value, (int, float, Interval)):
            matrix = IntervalMatrix.value_to_matrix_of_intervals(value)
            psd_interval = PsdIntervalInformation.value_to_psd_interval(value)
            return IntervalMatrixWithPsdInterval(matrix=matrix, psd_interval=psd_interval)
        if isinstance(value, PsdIntervalInformation):
            matrix = IntervalMatrix(shape=value.shape)
            return IntervalMatrixWithPsdInterval(matrix=matrix, psd_interval=value)
        if isinstance(value, IntervalMatrix):
            psd_interval = PsdIntervalInformation(shape=value.shape)
            return IntervalMatrixWithPsdInterval(matrix=value, psd_interval=psd_interval)
        raise ValueError(f'Unknown type of value: {type(value)}')

    @property
    def T(self):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix.T, psd_interval=self.interval)

    def __str__(self):
        return str(self.matrix) + '\nWith interval:\n' + str(self.interval)

    def __neg__(self):
        return IntervalMatrixWithPsdInterval(matrix=-self.matrix, psd_interval=-self.interval)

    def __add__(self, other):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix + other.matrix,
                                             psd_interval=self.interval + other.interval)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix - other.matrix,
                                             psd_interval=self.interval - other.interval)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix * other.matrix,
                                             psd_interval=self.interval * other.interval)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix ** power,
                                             psd_interval=self.interval ** power)

    def dot(self, other):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix.dot(other.matrix),
                                             psd_interval=self.interval.dot(other.interval))

    def sign(self):
        return self.interval.sign()

    def is_gershgorin_convex(self):
        return self.matrix.is_gershgorin_convex()

    @staticmethod
    def matrix_power(matrix_with_psd_interval, power):
        matrix = IntervalMatrix.matrix_power(matrix_with_psd_interval.matrix, power)
        psd_interval = PsdIntervalInformation.matrix_power(matrix_with_psd_interval.interval, power)
        return IntervalMatrixWithPsdInterval(matrix=matrix, psd_interval=psd_interval)
