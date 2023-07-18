import sympy as sym
from abc import abstractmethod
import interval
from intervalmatrix import IntervalMatrix


class BaseConvexDetector:
    def convexity_detection(self, expr, symbol_space=None, **kwargs):
        '''
        Returns True if convex
                False if concave
                None if neither or nothing can be determined
        '''
        if isinstance(expr, str):
            expr = self.parse_str(expr, **kwargs)
        return self._convexity_detection_expression(expr, symbol_space=symbol_space)

    def parse_str(self, string, **kwargs):
        msg = f'method \"parse_str\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def _convexity_detection_expression(self, expression, symbol_space=None):
        msg = f'method \"_is_convex_expression\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)


class DCPConvexDetector(BaseConvexDetector):
    def _convexity_detection_expression(self, expression, symbol_space=None):
        is_convex = expression.is_dcp()
        if is_convex:
            return True
        is_concave = (-expression).is_dcp()
        if is_concave:
            return False
        return None


class HessianConvexDetector(BaseConvexDetector):
    def parse_str(self, string, matrix_symbol_dict=None, **kwargs):
        if matrix_symbol_dict is None:
            return sym.parsing.sympy_parser.parse_expr(string, evaluate=False)
        return sym.parsing.sympy_parser.parse_expr(string, local_dict=matrix_symbol_dict, evaluate=False)

    def _convexity_detection_expression(self, expression, symbol_space=None):
        symbols = list(expression.free_symbols)
        if len(symbols) != 1:
            raise ValueError(f'Expression should have only one named variable (which can be matrix), got {symbols}')
        second_diff = expression.diff(symbols[0]).diff(symbols[0])
        value_interval = self._get_interval(second_diff, symbol_space=symbol_space)
        return value_interval.sign()

    @abstractmethod
    def _match_atomic(self, atomic_expr, symbol_space=None):
        if isinstance(atomic_expr, sym.core.numbers.Integer):
            return int(atomic_expr)
        if isinstance(atomic_expr, sym.core.symbol.Symbol):
            if (symbol_space is not None) and (str(atomic_expr) in symbol_space):
                return symbol_space[str(atomic_expr)]
            return interval.Interval([float('-inf'), float('inf')])
        if isinstance(atomic_expr, sym.matrices.expressions.matexpr.MatrixSymbol):
            if atomic_expr.args[2] != 1:
                raise NotImplementedError('Matrix is not a vector!')
            return IntervalMatrix(is_psd=None)
        if isinstance(atomic_expr, sym.Identity):
            return interval.Interval([1, 1])

        if isinstance(atomic_expr, sym.core.Mul):
            if len(atomic_expr.args) == 2:
                first_arg = atomic_expr.args[0]
                second_arg = atomic_expr.args[1]

                first_is_matrix = isinstance(first_arg, sym.matrices.expressions.MatrixExpr)
                second_is_matrix = isinstance(second_arg, sym.matrices.expressions.MatrixExpr)
                if first_is_matrix and second_is_matrix and first_arg.equals(second_arg.T):
                    return interval.Interval([0, float('inf')])
        return None

    @abstractmethod
    def _get_interval(self, expression, symbol_space=None):
        template_interval = self._match_atomic(expression, symbol_space=symbol_space)
        if template_interval is not None:
            return template_interval

        sub_intervals = []
        for sub_tree in expression.args:
            sub_intervals.append(self._get_interval(sub_tree, symbol_space=symbol_space))

        symbols = [sym.Symbol(f'x{i}') for i in range(len(sub_intervals))]
        root_expression = expression.func(*symbols)
        custom_modules = [
            {
            'sin': interval.Interval.sin,
            'cos': interval.Interval.cos,
            'exp': interval.Interval.exp,
            'ln': interval.Interval.ln,
            'log': interval.Interval.ln,
            },
            'numpy'
        ]

        root_expression_func = sym.utilities.lambdify(symbols, root_expression, modules=custom_modules)

        return root_expression_func(*sub_intervals)

