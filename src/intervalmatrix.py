from interval import Interval


class IntervalMatrix:
    # TODO: add self.data support
    def __init__(self, is_psd=None, psd_interval=None):
        # assert len(shape) == 2, f'Only matrices with two dimensions are supported, got shape = {shape}'
        # if values is None:
        #     assert shape is not None, 'Either shape or values shall be passed'
        #     self.data = [[Interval([float('-inf'), float('inf')]) for j in range(shape[1])] for i in range(shape[0])]
        # else:
        #     self.data = values

        if is_psd is None:
            if psd_interval is None:
                self.interval = Interval([float('-inf'), float('inf')])
            else:
                self.interval = psd_interval
        else:
            assert psd_interval is None, f'Only one of is_psd and psd_interval should be passed, got both'
            if is_psd:
                self.interval = Interval([0, float('inf')])
            else:
                self.interval = Interval([float('-inf'), 0])

    def __neg__(self):
        self.interval = -self.interval

    def __add__(self, other):
        return IntervalMatrix(psd_interval=self.interval .__add__(other))

    def __radd__(self, other):
        return IntervalMatrix(psd_interval=self.interval .__radd__(other))

    def __sub__(self, other):
        return IntervalMatrix(psd_interval=self.interval .__sub__(other))

    def __rsub__(self, other):
        return IntervalMatrix(psd_interval=self.interval .__rsub__(other))

    def __mul__(self, other):
        if isinstance(other, int | float):
            return IntervalMatrix(psd_interval=self.interval.__mul__(other))
        return IntervalMatrix(is_psd=None)

    def __rmul__(self, other):
        if isinstance(other, int | float):
            return IntervalMatrix(psd_interval=self.interval.__rmul__(other))
        return IntervalMatrix(is_psd=None)

    def __pow__(self, other):
        if other == 1:
            return self
        return IntervalMatrix(is_psd=None)
