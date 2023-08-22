from convex_detector import *

import sympy as sym
import numpy as np
import cvxpy

from interval import Interval


def TestDcp(dcp_convex_detector):
    x, y = cvxpy.Variable(), cvxpy.Variable()
    assert dcp_convex_detector.convexity_detection_expression((x**2))

    assert dcp_convex_detector.convexity_detection_expression(x * x) is None


def TestSingleVariable(convex_detector):
    assert convex_detector.convexity_detection_str('x * ln(x) + 1', symbol_values={'x': Interval([0, float('inf')])})
    assert convex_detector.convexity_detection_str('2 - x', symbol_values={'x': Interval([0, 1])})
    assert convex_detector.convexity_detection_str('x + sin(5 * x)', symbol_values={'x': Interval([0, 1])}) is None
    assert not convex_detector.convexity_detection_str('sin(x)', symbol_values={'x': Interval([0, 3.14])})
    print('Finished TestSingleVariable')

def TestSingleVariableInterval(convex_detector):
    x = sym.Symbol('x')
    assert convex_detector._get_matrix(2 - x, symbol_values={'x': Interval([0, 1])}).interval == Interval([1, 2])
    assert convex_detector._get_matrix(2 - x ** 2, symbol_values={'x': Interval([0, 1])}).interval == Interval([1, 2])
    assert convex_detector._get_matrix(2 - 2 * x, symbol_values={'x': Interval([0, 1])}).interval == Interval([0, 2])
    print('Finished TestSingleVariableInterval')


def TestMultiVariable(convex_detector):
    expr = 'X.T * X + 2 * I'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 5, 1), 'I': sym.Identity(1)}
    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict)

    expr = 'X.T * X'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 10, 1)}
    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict)

    expr = 'X.T * A * X'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 10, 1), 'A': sym.MatrixSymbol('A', 10, 10)}

    vectorized_to_interval = np.vectorize(Interval.valueToInterval)
    values = vectorized_to_interval(np.diag(np.arange(1, 11)).astype(object))

    symbol_space = {'A': convex_detector._default_psd(shape=(10, 10))}
    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict, symbol_values=symbol_space)

    symbol_space = {'A': convex_detector._default_substitution(shape=(10, 10))}
    assert convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict, symbol_values=symbol_space) is None

    print('Finished TestMultiVariable')


def TestMultiVariableInternals(convex_detector):
    pos_interval = Interval([0, float('inf')])

    X = sym.MatrixSymbol('X', 3, 1)
    func = X.T * X + 2 * sym.Identity(1)
    assert convex_detector.convexity_detection_expression(func)
    assert convex_detector._match_atomic(X.T * X) == pos_interval
    assert convex_detector.convexity_detection_str(func)

    Y = sym.MatrixSymbol('Y', 3, 3)
    interval_matrix = convex_detector._get_matrix(2 * Y, symbol_values={'Y': PsdIntervalInformation(shape=(3, 3), is_psd=True)})
    assert interval_matrix.interval == Interval([0, float('inf')])
    print('Finished TestMultiVariableInternals')


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


def TestAll():
    print('Test DCP')
    dcp_detector = DCPConvexDetector()
    TestDcp(dcp_detector)
    print('Finished')
    print()

    print('Testing PsdIntervalConvexDetector')
    psd_interval_convex_detector = PsdIntervalConvexDetector()
    try:
        TestSingleVariable(psd_interval_convex_detector)
        TestMultiVariable(psd_interval_convex_detector)
    except AssertionError as e:
        print(f'Failed tests!')
    print('Finished')
    print()

    print('Testing GershgorinConvexDetector')
    convex_detector = GershgorinConvexDetector()
    try:
        TestSingleVariable(convex_detector)
        TestMultiVariable(convex_detector)
    except AssertionError as e:
        print(f'Failed tests!')
    print('Finished')
    print()

    print('Testing CombinedConvexDetector')
    convex_detector = CombinedConvexDetector()
    try:
        TestSingleVariable(convex_detector)
        TestMultiVariable(convex_detector)
    except AssertionError as e:
        print(f'Failed tests!')
    print('Finished')
    print()

    try:
        TestGershgorin()
    except AssertionError as e:
        print(f'Failed!')

    print('Finished all!')


if __name__ == '__main__':
    combined_convex_detector = CombinedConvexDetector()

    expr = 'X.T * A * X'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 10, 1), 'A': sym.MatrixSymbol('A', 10, 10)}

    vectorized_to_interval = np.vectorize(Interval.valueToInterval)
    values = vectorized_to_interval(np.diag(np.arange(1, 11)).astype(object))
    symbol_space = {'A': IntervalMatrixWithPsdInterval(matrix=IntervalMatrix(values=values))}

    print(combined_convex_detector.convexity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict,
                                                           symbol_values=symbol_space))

    expr = 'HadamardPower(X, 10) * A'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 10, 1), 'A': sym.MatrixSymbol('A', 1, 10)}

    symbol_space = {'A': IntervalMatrixWithPsdInterval.full(shape=(1, 10), value=1)}

    print(combined_convex_detector._positivity_detection_str(expr, matrix_symbol_dict=matrix_symbol_dict,
                                                             symbol_values=symbol_space))
    #TestAll()
