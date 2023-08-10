from convex_detector import *

import sympy as sym
import numpy as np


def TestSingleVariable(convex_detector):
    assert convex_detector.convexity_detection_str('x * ln(x) + 1', symbol_space={'x': Interval([0, float('inf')])})
    assert convex_detector.convexity_detection_str('2 - x', symbol_space={'x': Interval([0, 1])})
    assert convex_detector.convexity_detection_str('x + sin(5 * x)', symbol_space={'x': Interval([0, 1])}) is None
    assert not convex_detector.convexity_detection_str('sin(x)', symbol_space={'x': Interval([0, 3.14])})


    x = sym.Symbol('x')
    assert convex_detector._get_interval(2 - x, symbol_space={'x': Interval([0, 1])}) == Interval([1, 2])
    assert convex_detector._get_interval(2 - x ** 2, symbol_space={'x': Interval([0, 1])}) == Interval([1, 2])
    assert convex_detector._get_interval(2 - 2 * x, symbol_space={'x': Interval([0, 1])}) == Interval([0, 2])
    print('Finished single variable')


def TestMultiVariable(convex_detector):
    expr = 'X.T * X + 2 * I'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 5, 1), 'I': sym.Identity(1)}
    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict)

    expr = 'X.T * X'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 10, 1)}
    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict)

    expr = 'X.T * A * X'
    vectorized_to_interval = np.vectorize(Interval.valueToInterval)
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 10, 1), 'A': sym.MatrixSymbol('A', 10, 10)}

    vectorized_to_interval = np.vectorize(Interval.valueToInterval)
    values = vectorized_to_interval(np.diag(np.arange(1, 11)).astype(object))
    symbol_space = {'A': IntervalMatrix(values=values)}

    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict, symbol_space=symbol_space) is None

    print('Finished multi variable')


def TestMultiVariableInternals(convex_detector):
    pos_interval = Interval([0, float('inf')])

    X = sym.MatrixSymbol('X', 3, 1)
    func = X.T * X + 2 * sym.Identity(1)
    assert convex_detector.convexity_detection_expression(func)
    assert convex_detector._match_atomic(X.T * X) == pos_interval
    assert convex_detector.convexity_detection_str(func)

    Y = sym.MatrixSymbol('Y', 3, 3)
    interval_matrix = convex_detector._get_interval(2 * Y, symbol_space={'Y': IntervalMatrix(shape=(3, 3), is_psd=True)})
    assert interval_matrix.interval == Interval([0, float('inf')])
    print('Finished multi variable Internals')


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
    TestMultiVariableInternals(hessian_convex_detector)

    TestGershgorin()

    print('Finished')
