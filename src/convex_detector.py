import sympy as sym
from interval import Interval
from intervalmatrix import IntervalMatrix, matrix_power
from psd_interval_information import PsdIntervalInformation


class BaseConvexDetector:
    def convexity_detection_str(self, expr, symbol_values=None, **kwargs):
        """
        Returns
            True if convex
            False if concave
            None if neither or nothing can be determined
        """

        if isinstance(expr, str):
            expr = self.parse_str(expr, **kwargs)
        return self.convexity_detection_expression(expr, symbol_values=symbol_values)

    def parse_str(self, string, **kwargs):
        msg = f'method \"parse_str\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def convexity_detection_expression(self, expression, symbol_values=None):
        msg = f'method \"_is_convex_expression\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)


class DCPConvexDetector(BaseConvexDetector):
    def convexity_detection_expression(self, expression, symbol_values=None):
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

    def convexity_detection_expression(self, expression, symbol_values=None):
        symbols = expression.free_symbols
        probably_variables = {'x', 'X', 'y', 'Y', 'z', 'Z'}
        variables = list(set(map(str, symbols)) & probably_variables)

        if len(variables) != 1:
            msg = f'Expression should have only one named variable (which can be matrix), detected {symbols}'
            raise ValueError(msg)
        for symbol in symbols:
            if str(symbol) == variables[0]:
                variable = symbol
                break
        else:
            raise ValueError(f'Can not detect variable')
        second_diff = sym.diff(sym.diff(expression, variable), variable)
        return self._positivity_detection(second_diff, symbol_values)

    def _positivity_detection(self, expression, symbol_values=None):
        msg = f'method \"_positivity_detection\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)


class PsdIntervalConvexDetector(HessianConvexDetector):
    def _positivity_detection(self, expression, symbol_values=None):
        value_interval = self._get_interval(expression, symbol_values=symbol_values)
        return value_interval.sign()

    @staticmethod
    def _match_atomic(atomic_expr, symbol_values=None):
        if isinstance(atomic_expr, sym.core.symbol.Symbol):
            if (symbol_values is not None) and (str(atomic_expr) in symbol_values):
                return PsdIntervalInformation.value_to_psd_interval(symbol_values[str(atomic_expr)])
            return PsdIntervalInformation(shape=None, interval=None)
        
        if isinstance(atomic_expr, sym.core.numbers.Integer):
            return PsdIntervalInformation.value_to_psd_interval(int(atomic_expr))
        if isinstance(atomic_expr, sym.core.numbers.Float):
            return PsdIntervalInformation.value_to_psd_interval(float(atomic_expr))

        if isinstance(atomic_expr, sym.matrices.expressions.matexpr.MatrixSymbol):
            if (symbol_values is not None) and (str(atomic_expr) in symbol_values):
                return PsdIntervalInformation.value_to_psd_interval(symbol_values[str(atomic_expr)])
            return PsdIntervalInformation(shape=atomic_expr.shape, interval=None)
        if isinstance(atomic_expr, sym.Identity):
            return PsdIntervalInformation(shape=atomic_expr.shape, is_psd=True)

        if isinstance(atomic_expr, sym.core.Mul):
            if len(atomic_expr.args) == 2:
                first_arg = atomic_expr.args[0]
                second_arg = atomic_expr.args[1]

                first_is_matrix = isinstance(first_arg, sym.matrices.expressions.MatrixExpr)
                second_is_matrix = isinstance(second_arg, sym.matrices.expressions.MatrixExpr)
                if first_is_matrix and second_is_matrix and first_arg.equals(second_arg.T):
                    shape = (first_arg.shape[0], second_arg.shape[1])
                    return PsdIntervalInformation(shape=shape, is_psd=True)
        return None

    def _get_interval(self, expression, symbol_values=None):
        template_interval = self._match_atomic(expression, symbol_values=symbol_values)
        if template_interval is not None:
            return template_interval

        sub_intervals = []
        for sub_tree in expression.args:
            sub_intervals.append(self._get_interval(sub_tree, symbol_values=symbol_values))

        symbols = []
        for i, sub_interval in enumerate(sub_intervals):
            if sub_interval.is_scalar():
                symbols.append(sym.Symbol(f'x{i}'))
            else:
                symbols.append(sym.MatrixSymbol(f'X{i}', sub_interval.shape[0], sub_interval.shape[1]))

        root_expression = expression.func(*symbols)
        custom_modules = [
            {
                'sin': PsdIntervalInformation.sin,
                'cos': PsdIntervalInformation.cos,
                'exp': PsdIntervalInformation.exp,
                'ln': PsdIntervalInformation.ln,
                'log': PsdIntervalInformation.ln,
                'matrix_power': PsdIntervalInformation.matrix_power
            },
            'numpy'
        ]
        root_expression_func = sym.utilities.lambdify(symbols, root_expression, modules=custom_modules)
        return root_expression_func(*sub_intervals)


class GershgorinConvexDetector(HessianConvexDetector):
    pass


class CombinedConvexDetector(HessianConvexDetector):
    def _positivity_detection(self, expression, symbol_values=None):
        value_interval = self._get_interval(expression, symbol_values=symbol_values)
        return value_interval.sign()

    @staticmethod
    def _match_atomic(atomic_expr, symbol_values=None):
        if isinstance(atomic_expr, sym.core.numbers.Integer):
            return Interval.valueToInterval(int(atomic_expr))
        if isinstance(atomic_expr, sym.core.numbers.Float):
            return Interval.valueToInterval(float(atomic_expr))
        if isinstance(atomic_expr, sym.core.symbol.Symbol):
            if (symbol_values is not None) and (str(atomic_expr) in symbol_values):
                return symbol_values[str(atomic_expr)]
            return Interval([float('-inf'), float('inf')])
        if isinstance(atomic_expr, sym.matrices.expressions.matexpr.MatrixSymbol):
            if (symbol_values is not None) and (str(atomic_expr) in symbol_values):
                return symbol_values[str(atomic_expr)]
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

    def _get_interval(self, expression, symbol_values=None):
        template_interval = self._match_atomic(expression, symbol_values=symbol_values)
        if template_interval is not None:
            return template_interval

        sub_intervals = []
        for sub_tree in expression.args:
            sub_intervals.append(self._get_interval(sub_tree, symbol_values=symbol_values))

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
