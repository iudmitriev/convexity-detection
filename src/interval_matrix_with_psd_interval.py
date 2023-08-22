from interval import Interval
from interval_matrix import IntervalMatrix
from psd_interval_information import PsdIntervalInformation


class IntervalMatrixWithPsdInterval:
    """
    Combines classes IntervalMatrix and PsdIntervalInformation, and has two parts:
    1) Matrix, each element of which is an interval
    2) Interval, that encodes psd information about a scalar, vector or matrix:
        For scalar expressions, the interval encodes (a superset of) the domain of the expression
        For vector expressions, the interval encodes (a superset of) the domains of all vector entries
        For matrix expression, any interval in [0, inf) encodes psd information,
                               any interval in (−inf, 0] encodes nsd information
                               other intervals encodes no information

    Attributes:
        matrix: matrix of intervals
        interval: interval, that encodes psd information
    """
    def __init__(self, matrix=None, psd_interval=None):
        """
        Parameters:
            matrix: IntervalMatrix
                matrix of intervals
                if None, default value of IntervalMatrix is used
            psd_interval: PsdIntervalInformation
                interval, that encodes psd information
                If None, default value of PsdIntervalInformation is used
        """
        if matrix is None:
            matrix = IntervalMatrix()
        self.matrix = matrix
        if psd_interval is None:
            psd_interval = PsdIntervalInformation()
        self.interval = psd_interval

        is_gershgorin_psd = self.matrix.is_gershgorin_convex()
        if is_gershgorin_psd is not None and self.interval.sign() is None:
            self.interval = PsdIntervalInformation(is_psd=is_gershgorin_psd)

    @property
    def shape(self):
        """
        Returns:
            shape of the corresponding matrix
        """
        return self.matrix.shape

    @staticmethod
    def value_to_interval_matrix_with_psd_interval(value):
        """
                Converts value to IntervalMatrix
                Parameters:
                    value: int, float, Interval, IntervalMatrix,
                           PsdIntervalInformation or IntervalMatrixWithPsdInterval
                        value to be converted
                Returns:
                    Converted value
                """
        if isinstance(value, IntervalMatrixWithPsdInterval):
            return value
        if isinstance(value, (int, float, Interval)):
            matrix = IntervalMatrix.value_to_interval_matrix(value)
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

    def is_scalar(self):
        """
        Returns:
            True if object is scalar
            False otherwise
        """
        return self.matrix.is_scalar()

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
        return IntervalMatrixWithPsdInterval(matrix=self.matrix ** power.matrix,
                                             psd_interval=self.interval ** power.interval)

    def dot(self, other):
        return IntervalMatrixWithPsdInterval(matrix=self.matrix.dot(other.matrix),
                                             psd_interval=self.interval.dot(other.interval))

    def sign(self):
        """
        Returns
            True if object is psd
            False if object is nsd
            None if none of the above
        """
        return self.interval.sign()

    def is_gershgorin_convex(self):
        """
        Calculates lower and upper bounds for matrix of intervals
        using gershgorin theorem and gets corresponding convex information

        Returns:
            True if object is psd
            False if object is nsd
            None if none of the above
        """
        return self.matrix.is_gershgorin_convex()

    @staticmethod
    def matrix_power(matrix_with_psd_interval, power):
        matrix = IntervalMatrix.matrix_power(matrix_with_psd_interval.matrix, power)
        psd_interval = PsdIntervalInformation.matrix_power(matrix_with_psd_interval.interval, power)
        return IntervalMatrixWithPsdInterval(matrix=matrix, psd_interval=psd_interval)

    @staticmethod
    def sin(matrix_with_psd_interval):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix.sin(matrix_with_psd_interval.matrix),
                                             psd_interval=PsdIntervalInformation.sin(matrix_with_psd_interval.interval))

    @staticmethod
    def cos(matrix_with_psd_interval):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix.cos(matrix_with_psd_interval.matrix),
                                             psd_interval=PsdIntervalInformation.cos(matrix_with_psd_interval.interval))

    @staticmethod
    def exp(matrix_with_psd_interval):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix.exp(matrix_with_psd_interval.matrix),
                                             psd_interval=PsdIntervalInformation.exp(matrix_with_psd_interval.interval))

    @staticmethod
    def ln(matrix_with_psd_interval):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix.ln(matrix_with_psd_interval.matrix),
                                             psd_interval=PsdIntervalInformation.ln(matrix_with_psd_interval.interval))

    @staticmethod
    def log(matrix_with_psd_interval):
        return IntervalMatrixWithPsdInterval.ln(matrix_with_psd_interval)
