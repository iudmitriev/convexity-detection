from convex_detector import *

import sympy as sym

if __name__ == '__main__':
    convex_detector = HessianConvexDetector()
    assert convex_detector.convexity_detection('x * ln(x) + 1', symbol_space={'x': interval.Interval([0, float('inf')])})

    X = sym.MatrixSymbol('X', 3, 1)
    func = X.T * X + 2 * sym.Identity(1)
    assert convex_detector._convexity_detection_expression(func)
    assert convex_detector._match_atomic(X.T * X) == interval.Interval([0, float('inf')])

    expr = 'X.T * X + 2 * I'
    matrix_symbol_dict = {'X': sym.MatrixSymbol('X', 5, 1), 'I': sym.Identity(1)}
    assert convex_detector.convexity_detection(expr, matrix_symbol_dict=matrix_symbol_dict)
    print('Finished')
