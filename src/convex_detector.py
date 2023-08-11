import sympy as sym
from abc import abstractmethod
from interval import Interval
from intervalmatrix import IntervalMatrix, matrix_power


class BaseConvexDetector:
    def convexity_detection_str(self, expr, symbol_space=None, **kwargs):
        '''
        Returns True if convex
                False if concave
                None if neither or nothing can be determined
        '''
        if isinstance(expr, str):
            expr = self.parse_str(expr, **kwargs)
        return self.convexity_detection_expression(expr, symbol_space=symbol_space)

    def parse_str(self, string, **kwargs):
        msg = f'method \"parse_str\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def convexity_detection_expression(self, expression, symbol_space=None):
        msg = f'method \"_is_convex_expression\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)


class DCPConvexDetector(BaseConvexDetector):
    def convexity_detection_expression(self, expression, symbol_space=None):
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

    def convexity_detection_expression(self, expression, symbol_space=None):
        symbols = expression.free_symbols
        probably_variables = {'x', 'X', 'y', 'Y', 'z', 'Z'}
        variables = list(set(map(str, symbols)) & probably_variables)

        if len(variables) != 1:
            raise ValueError(f'Expression should have only one named variable (which can be matrix), detected {symbols}')
        for symbol in symbols:
            if str(symbol) == variables[0]:
                variable = symbol
                break
        else:
            raise ValueError(f'Can not detect variable')
        second_diff = sym.diff(sym.diff(expression, variable), variable)
        return self._positivity_detection(second_diff, symbol_space)

    def _positivity_detection(self, expression, symbol_space=None):
        value_interval = self._get_interval(expression, symbol_space=symbol_space)
        return value_interval.sign()

    @abstractmethod
    def _match_atomic(self, atomic_expr, symbol_space=None):
        if isinstance(atomic_expr, sym.core.numbers.Integer):
            return Interval.valueToInterval(int(atomic_expr))
        if isinstance(atomic_expr, sym.core.numbers.Float):
            return Interval.valueToInterval(float(atomic_expr))
        if isinstance(atomic_expr, sym.core.symbol.Symbol):
            if (symbol_space is not None) and (str(atomic_expr) in symbol_space):
                return symbol_space[str(atomic_expr)]
            return Interval([float('-inf'), float('inf')])
        if isinstance(atomic_expr, sym.matrices.expressions.matexpr.MatrixSymbol):
            if (symbol_space is not None) and (str(atomic_expr) in symbol_space):
                return symbol_space[str(atomic_expr)]
            return IntervalMatrix(is_psd=None)
        if isinstance(atomic_expr, sym.Identity):
            return Interval([1, 1])

        if isinstance(atomic_expr, sym.core.Mul):
            if len(atomic_expr.args) == 2:
                first_arg = atomic_expr.args[0]
                second_arg = atomic_expr.args[1]

                first_is_matrix = isinstance(first_arg, sym.matrices.expressions.MatrixExpr)
                second_is_matrix = isinstance(second_arg, sym.matrices.expressions.MatrixExpr)
                if first_is_matrix and second_is_matrix and first_arg.equals(second_arg.T):
                    return Interval([0, float('inf')])
        return None

    @abstractmethod
    def _get_interval(self, expression, symbol_space=None):
        template_interval = self._match_atomic(expression, symbol_space=symbol_space)
        if template_interval is not None:
            return template_interval

        sub_intervals = []
        for sub_tree in expression.args:
            sub_intervals.append(self._get_interval(sub_tree, symbol_space=symbol_space))

        symbols = []
        for i, sub_interval in enumerate(sub_intervals):
            if isinstance(sub_interval, (int, float, Interval)):
                symbols.append(sym.Symbol(f'x{i}'))
            elif isinstance(sub_interval, IntervalMatrix):
                symbols.append(sym.MatrixSymbol(f'X{i}', sub_interval.shape[0], sub_interval.shape[1]))
        root_expression = expression.func(*symbols)
        custom_modules = [
            {
            'sin': Interval.sin,
            'cos': Interval.cos,
            'exp': Interval.exp,
            'ln': Interval.ln,
            'log': Interval.ln,
            'matrix_power': matrix_power
            },
            'numpy'
        ]

        root_expression_func = sym.utilities.lambdify(symbols, root_expression, modules=custom_modules)

        return root_expression_func(*sub_intervals)

