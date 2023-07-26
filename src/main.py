from convex_detector import *

import sympy as sym
import numpy as np


def TestSingleVariable(convex_detector):
    assert convex_detector.convexity_detection('x * ln(x) + 1', symbol_space={'x': Interval([0, float('inf')])})
    assert convex_detector.convexity_detection('2 - x', symbol_space={'x': Interval([0, 1])})

    x = sym.Symbol('x')
    assert convex_detector._get_interval(2 - x, symbol_space={'x': Interval([0, 1])}) == Interval([1, 2])
    assert convex_detector._get_interval(2 - x ** 2, symbol_space={'x': Interval([0, 1])}) == Interval([1, 2])
    assert convex_detector._get_interval(2 - 2 * x, symbol_space={'x': Interval([0, 1])}) == Interval([0, 2])
    print('Finished single variable')


def TestMultiVariable(convex_detector):
    pos_interval = Interval([0, float('inf')])

    X = sym.MatrixSymbol('X', 3, 1)
    func = X.T * X + 2 * sym.Identity(1)
    assert convex_detector._convexity_detection_expression(func)
    assert convex_detector._match_atomic(X.T * X) == pos_interval
    assert convex_detector.convexity_detection(func)

    Y = sym.MatrixSymbol('Y', 3, 3)
    interval_matrix = convex_detector._get_interval(2 * Y, symbol_space={'Y': IntervalMatrix(shape=(3, 3), is_psd=True)})
    assert interval_matrix.interval == Interval([0, float('inf')])

    expr = 'X.T * X + 2 * I'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 5, 1), 'I': sym.Identity(1)}
    assert convex_detector.convexity_detection(expr, matrix_symbol_dict=matrix_symbol_dict)
    print('Finished multi variable')


def TestGershgorin():
    values = np.array([
        [Interval([1, 1]), Interval([1, 1])],
        [Interval([0, 0]), Interval([1, 1])]
    ])
    assert IntervalMatrix(values=values).is_gershgorin_convex()

    values = np.array([
        [Interval([-1, -1]), Interval([-1, -1])],
        [Interval([0, 0]), Interval([-1, -1])]
    ])
    assert not IntervalMatrix(values=values).is_gershgorin_convex()

    values = np.array([
        [Interval([1, 1]), Interval([0, 0])],
        [Interval([0, 0]), Interval([-1, -1])]
    ])
    assert IntervalMatrix(values=values).is_gershgorin_convex() is None
    print('Finished Gershgorin')


if __name__ == '__main__':
    hessian_convex_detector = HessianConvexDetector()

    TestSingleVariable(hessian_convex_detector)
    TestMultiVariable(hessian_convex_detector)

    TestGershgorin()

    print('Finished')
