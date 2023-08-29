from interval import Interval


class PsdIntervalInformation:
    """
    Encodes psd information about a scalar, vector or matrix as described in "Convexity Certificates from Hessians":
    For scalar expressions, the interval encodes (a superset of) the domain of the expression
    For vector expressions, the interval encodes (a superset of) the domains of all vector entries
    For matrix expression, any interval in [0, inf) encodes psd information,
                           any interval in (−inf, 0] encodes nsd information
                           other intervals encodes no information

    Attributes:
        interval: psd information as described above
        shape: shape of the described object
    """
    def __init__(self, shape=None, interval=None, is_psd=None):
        """
        Parameters:
            shape: tuple or None
                shape of the described object, if None, (1, ) will be used
            interval: Interval or None
                psd information about the object
            is_psd: Bool or None
                shortcut for psd information about the object, should only be passed if interval is None
                if is_psd == True, Interval([0, float('inf')]) will be used
                if is_psd == False, Interval([float('-inf'), 0]) will be used
        """
        if shape is None:
            self.shape = (1, 1)
        else:
            if len(shape) >= 3:
                raise ValueError(f'Only 1d or 2d shapes are supported, got {shape}')
            if len(shape) == 1:
                shape += (1,)
            self.shape = shape

        if interval is None:
            if is_psd is None:
                self.interval = Interval([float('-inf'), float('inf')])
            elif is_psd:
                self.interval = Interval([0, float('inf')])
            else:
                self.interval = Interval([float('-inf'), 0])
        else:
            if is_psd is not None:
                msg = f'For PsdIntervalInformation only one of is_psd and psd_interval should be passed, got both'
                raise ValueError(msg)
            self.interval = interval

    def is_scalar(self):
        """
        Returns:
            True if object is scalar
            False otherwise
        """
        return self.shape == (1, 1)

    def is_vector(self):
        """
        Returns:
            True if object is vector
            False otherwise
        """
        return self.shape[0] == 1 or self.shape[1] == 1

    @staticmethod
    def value_to_psd_interval(value):
        """
        Converts value to PsdIntervalInformation
        Parameters:
            value: int, float, Interval, or PsdIntervalInformation
                value to be converted
        Returns:
            Converted value
        """
        if isinstance(value, PsdIntervalInformation):
            return value
        if isinstance(value, Interval):
            return PsdIntervalInformation(shape=(1, 1), interval=value)
        if isinstance(value, (int, float)):
            return PsdIntervalInformation(shape=(1, 1), interval=Interval.valueToInterval(value))

    @property
    def T(self):
        return PsdIntervalInformation(shape=(self.shape[1], self.shape[0]), interval=self.interval)

    def __str__(self):
        return f'PsdIntervalInformation(shape = {self.shape}, interval = {self.interval})'

    def __neg__(self):
        return PsdIntervalInformation(shape=self.shape, interval=-self.interval)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError(f'Shape mismatch in PsdIntervalInformation.__add__, got {self.shape} and {other.shape}')
        interval = self.interval.__add__(other.interval)
        return PsdIntervalInformation(shape=self.shape, interval=interval)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError(f'Shape mismatch in PsdIntervalInformation.__sub__, got {self.shape} and {other.shape}')
        interval = self.interval.__sub__(other.interval)
        return PsdIntervalInformation(shape=self.shape, interval=interval)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        if self.is_scalar():
            return PsdIntervalInformation(shape=other.shape, interval=self.interval * other.interval)
        elif other.is_scalar():
            return PsdIntervalInformation(shape=self.shape, interval=self.interval * other.interval)

        if self.shape != other.shape:
            raise ValueError(f'Shape mismatch in PsdIntervalInformation.__mul__, got {self.shape} and {other.shape}')

        if self.is_vector():
            # Two vectors
            return PsdIntervalInformation(shape=self.shape, interval=self.interval * other.interval)

        # Two matrices
        return PsdIntervalInformation(shape=self.shape, interval=None, is_psd=None)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        if self.is_vector():
            return PsdIntervalInformation(shape=self.shape, interval=self.interval ** power.interval)
        return PsdIntervalInformation(shape=self.shape, interval=None, is_psd=None)

    def __abs__(self):
        if self.is_vector():
            return PsdIntervalInformation(shape=self.shape, interval=abs(self.interval))
        return PsdIntervalInformation(shape=self.shape, interval=None, is_psd=None)

    @staticmethod
    def sin(psd_interval):
        if psd_interval.is_vector():
            return PsdIntervalInformation(shape=psd_interval.shape, interval=Interval.sin(psd_interval.interval))
        return PsdIntervalInformation(shape=psd_interval.shape, interval=None)

    @staticmethod
    def cos(psd_interval):
        if psd_interval.is_vector():
            return PsdIntervalInformation(shape=psd_interval.shape, interval=Interval.cos(psd_interval.interval))
        return PsdIntervalInformation(shape=psd_interval.shape, interval=None)

    @staticmethod
    def exp(psd_interval):
        if psd_interval.is_vector():
            return PsdIntervalInformation(shape=psd_interval.shape, interval=Interval.exp(psd_interval.interval))
        return PsdIntervalInformation(shape=psd_interval.shape, interval=None)

    @staticmethod
    def ln(psd_interval):
        if psd_interval.is_vector():
            return PsdIntervalInformation(shape=psd_interval.shape, interval=Interval.ln(psd_interval.interval))
        return PsdIntervalInformation(shape=psd_interval.shape, interval=None)

    @staticmethod
    def log(psd_interval):
        return PsdIntervalInformation.ln(psd_interval)

    def dot(self, other):
        """
        Matrix multiplication
        """
        if self.is_scalar() or other.is_scalar():
            return self * other

        if self.shape[1] != other.shape[0]:
            raise ValueError(f'Shape mismatch in PsdIntervalInformation.dot, got {self.shape} and {other.shape}')

        shape = (self.shape[0], other.shape[1])
        if shape == (1, 1):
            interval = self.interval * other.interval * self.shape[1]
            return PsdIntervalInformation(shape=self.shape, interval=interval)
        return PsdIntervalInformation(shape=self.shape, interval=None, is_psd=None)

    def sign(self):
        """
        Returns
            True if object is psd
            False if object is nsd
            None if none of the above
        """
        if self.interval.isIn(Interval([0, float('inf')])):
            return True
        if self.interval.isIn(Interval([float('-inf'), 0])):
            return False
        return None

    @staticmethod
    def matrix_power(psd_interval, power):
        if not power.is_scalar():
            raise ValueError('Only scalar powers are supported')

        if psd_interval.is_scalar():
            return PsdIntervalInformation(shape=psd_interval.shape, interval=psd_interval.interval ** power.interval)

        if psd_interval.shape[0] != psd_interval.shape[1]:
            msg = f'Matrix should be square in PsdIntervalInformation.matrix_power, got {psd_interval.shape}'
            raise ValueError(msg)
        return PsdIntervalInformation(shape=psd_interval.shape, interval=None)

    def __eq__(self, other):
        other = PsdIntervalInformation.value_to_psd_interval(other)
        return self.shape == other.shape and self.interval == other.interval

    @staticmethod
    def eye(shape):
        """
        Returns:
            Identity matrix of size shape
        """
        return PsdIntervalInformation(shape=shape, is_psd=True)

    @staticmethod
    def zero(shape):
        """
        Returns:
            Zero matrix of size shape
        """
        return PsdIntervalInformation(shape=shape, is_psd=True)

    @staticmethod
    def full(value, shape):
        """
        Return a new matrix of given shape, filled with value.
        Parameters:
            value: int, float or Interval
                Value to fill
            shape: tuple
                Shape of the new matrix
        Returns:
            matrix: IntervalMatrix
                Matrix of given shape, filled with value
        """
        return PsdIntervalInformation(shape=shape)
