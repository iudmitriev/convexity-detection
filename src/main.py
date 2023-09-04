from convex_detector import *

import sympy as sym
import numpy as np
import cvxpy

from interval import Interval

import time


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


def Test1():
    function = 'X.T * X'
    print(function)
    for method, convex_detector in methods.items():
        n_runs = 100
        result = -1
        times = []
        for i in range(n_runs):
            start = time.time()
            result = convex_detector.convexity_detection_str(function, matrix_symbol_dict=matrix_symbol_dict)
            end = time.time()
            times.append(end - start)
        times = np.array(times)
        print(f'{method}: {result}, {times.mean():.5f}s \\pm {times.std():.5f}s')



def Test2():
    function = 'X.T * N * X'
    print(function)
    vectorized_to_interval = np.vectorize(Interval.valueToInterval)
    values = vectorized_to_interval(np.diag(np.arange(1, 11)).astype(object))
    for method, convex_detector in methods.items():

        if method == 'PsdInterval':
            N = convex_detector._default_substitution(shape=(10, 10))
        else:
            N = convex_detector._value_to_substitution(IntervalMatrix(values=values))

        symbol_space = {
            'N': N
        }

        n_runs = 100
        times = []
        result = -1
        for i in range(n_runs):
            start = time.time()
            result = convex_detector.convexity_detection_str(function, matrix_symbol_dict=matrix_symbol_dict,
                                                             symbol_values=symbol_space)
            end = time.time()
            times.append(end - start)
        times = np.array(times)
        print(f'{method}: {result}, {times.mean():.5f}s \\pm {times.std():.5f}s')


def Test3():
    function = '(HadamardPower(X, 10)).T * E'
    print(function)
    vectorized_to_interval = np.vectorize(Interval.valueToInterval)
    values = vectorized_to_interval(np.diag(np.arange(1, 11)).astype(object))
    for method, convex_detector in methods.items():

        if method == 'PsdInterval':
            E = PsdIntervalInformation.full(value=1, shape=(10, 1))
        elif method == 'Gershgorin':
            E = IntervalMatrix.full(value=1, shape=(10, 1))
        else:
            E = IntervalMatrixWithPsdInterval.full(value=1, shape=(10, 1))

        symbol_space = {
            'E': E
        }

        n_runs = 2
        times = []
        result = -1
        try:
            for i in range(n_runs):
                start = time.time()
                result = convex_detector.convexity_detection_str(function, matrix_symbol_dict=matrix_symbol_dict,
                                                                 symbol_values=symbol_space)
                end = time.time()
                times.append(end - start)
            times = np.array(times)
            print(f'{method}: {result}, {times.mean():.5f}s \\pm {times.std():.5f}s')
        except Exception as e:
            print(f'Failed with error {e}')


def Test4():
    function = '-exp(-0.5 * X.T * X)'
    print(function)
    for method, convex_detector in methods.items():
        n_runs = 1
        result = -1
        times = []
        for i in range(n_runs):
            start = time.time()
            result = convex_detector.convexity_detection_str(function, matrix_symbol_dict=matrix_symbol_dict)
            end = time.time()
            times.append(end - start)
        times = np.array(times)
        print(f'{method}: {result}, {times.mean():.5f}s \\pm {times.std():.5f}s')


if __name__ == '__main__':
    matrix_symbol_dict = {
        'X': sym.MatrixSymbol('X', 10, 1),
        'E': sym.MatrixSymbol('E', 10, 1),
        'N': sym.MatrixSymbol('N', 10, 10)
    }

    # list_of_functions_to_test = [
    #   'X.T * X',
    #   'X.T * N * X',
    #   '(HadamardPower(X, 10)).T * E',
    #   '-exp(-0.5 * X.T * X)'
    # ]

    methods = {
        'PsdInterval': PsdIntervalConvexDetector(),
        'Gershgorin': GershgorinConvexDetector(),
        'Combined': CombinedConvexDetector()
    }

    # Test1()
    # Test2()
    # Test3()
    Test4()
