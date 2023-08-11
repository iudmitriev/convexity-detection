import sympy as sym

from interval_matrix import IntervalMatrix
from psd_interval_information import PsdIntervalInformation
from interval_matrix_with_psd_interval import IntervalMatrixWithPsdInterval

from abc import ABC, abstractmethod


class BaseConvexDetector(ABC):
    def convexity_detection_str(self, expr, symbol_values=None, **kwargs):
        """
        Returns
            True if function is convex
            False if function is concave
            None if function is none of the above or nothing can be determined
        """

        if isinstance(expr, str):
            expr = self.parse_str(expr, **kwargs)
        return self.convexity_detection_expression(expr, symbol_values=symbol_values)

    @abstractmethod
    def parse_str(self, string, **kwargs):
        pass

    @abstractmethod
    def convexity_detection_expression(self, expression, symbol_values=None):
        pass


class DCPConvexDetector(BaseConvexDetector):
    def parse_str(self, string, **kwargs):
        raise NotImplementedError('No CVXPY parser from string have been found')

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
        second_diff = self._get_second_diff(expression)
        return self._positivity_detection(second_diff, symbol_values)

    @staticmethod
    def _get_second_diff(expression):
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
        return sym.diff(sym.diff(expression, variable), variable)

    @abstractmethod
    def _positivity_detection(self, expression, symbol_values=None):
        pass


class SubstitutingHessianConvexDetector(HessianConvexDetector):
    @property
    def custom_modules(self):
        msg = f'method \"custom_modules\" not implemented for SubstitutingHessianConvexDetector'
        raise NotImplementedError(msg)

    def _positivity_detection(self, expression, symbol_values=None):
        substitution = self._get_substitution(expression, symbol_values=symbol_values)
        return substitution.sign()

    @staticmethod
    @abstractmethod
    def _value_to_substitution(value):
        pass

    @staticmethod
    @abstractmethod
    def _default_substitution(shape=None):
        pass

    @staticmethod
    @abstractmethod
    def _default_psd(shape=None):
        pass

    @staticmethod
    @abstractmethod
    def _identity(shape=None):
        pass

    def _match_atomic(self, atomic_expr, symbol_values=None):

        if isinstance(atomic_expr, sym.core.symbol.Symbol):
            if (symbol_values is not None) and (str(atomic_expr) in symbol_values):
                return self._value_to_substitution(symbol_values[str(atomic_expr)])
            return self._default_substitution()

        if isinstance(atomic_expr, sym.core.numbers.Integer):
            return self._value_to_substitution(int(atomic_expr))
        if isinstance(atomic_expr, sym.core.numbers.Float):
            return self._value_to_substitution(float(atomic_expr))

        if isinstance(atomic_expr, sym.matrices.expressions.matexpr.MatrixSymbol):
            if (symbol_values is not None) and (str(atomic_expr) in symbol_values):
                return self._value_to_substitution(symbol_values[str(atomic_expr)])
            return self._default_substitution(shape=atomic_expr.shape)
        if isinstance(atomic_expr, sym.Identity):
            return self._identity(shape=atomic_expr.shape)

        if isinstance(atomic_expr, sym.core.Mul):
            if len(atomic_expr.args) == 2:
                first_arg = atomic_expr.args[0]
                second_arg = atomic_expr.args[1]

                first_is_matrix = isinstance(first_arg, sym.matrices.expressions.MatrixExpr)
                second_is_matrix = isinstance(second_arg, sym.matrices.expressions.MatrixExpr)
                if first_is_matrix and second_is_matrix and first_arg.equals(second_arg.T):
                    shape = (first_arg.shape[0], second_arg.shape[1])
                    return self._default_psd(shape=shape)
        return None

    def _get_substitution(self, expression, symbol_values=None):
        template_interval = self._match_atomic(expression, symbol_values=symbol_values)
        if template_interval is not None:
            return template_interval

        sub_intervals = []
        for sub_tree in expression.args:
            sub_intervals.append(self._get_substitution(sub_tree, symbol_values=symbol_values))

        symbols = []
        for i, sub_interval in enumerate(sub_intervals):
            if sub_interval.is_scalar():
                symbols.append(sym.Symbol(f'x{i}'))
            else:
                symbols.append(sym.MatrixSymbol(f'X{i}', sub_interval.shape[0], sub_interval.shape[1]))

        root_expression = expression.func(*symbols)
        root_expression_func = sym.utilities.lambdify(symbols, root_expression, modules=self.custom_modules)
        return root_expression_func(*sub_intervals)


class PsdIntervalConvexDetector(SubstitutingHessianConvexDetector):
    @property
    def custom_modules(self):
        return [
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

    @staticmethod
    def _value_to_substitution(value):
        return PsdIntervalInformation.value_to_psd_interval(value)

    @staticmethod
    def _default_substitution(shape=None):
        return PsdIntervalInformation(shape=shape)

    @staticmethod
    def _default_psd(shape=None):
        return PsdIntervalInformation(shape=shape, is_psd=True)

    @staticmethod
    def _identity(shape=None):
        return PsdIntervalInformation(shape=shape, is_psd=True)


class GershgorinConvexDetector(SubstitutingHessianConvexDetector):
    @property
    def custom_modules(self):
        return [
            {
                'sin': IntervalMatrix.sin,
                'cos': IntervalMatrix.cos,
                'exp': IntervalMatrix.exp,
                'ln': IntervalMatrix.ln,
                'log': IntervalMatrix.ln,
                'matrix_power': IntervalMatrix.matrix_power
            },
            'numpy'
        ]

    @staticmethod
    def _value_to_substitution(value):
        return IntervalMatrix.value_to_interval_matrix(value)

    @staticmethod
    def _default_substitution(shape=None):
        return IntervalMatrix(shape=shape)

    @staticmethod
    def _default_psd(shape=None):
        return IntervalMatrix(shape=shape)

    @staticmethod
    def _identity(shape=None):
        return IntervalMatrix.eye(shape[0])


class CombinedConvexDetector(SubstitutingHessianConvexDetector):
    @property
    def custom_modules(self):
        return [
            {
                'sin': IntervalMatrixWithPsdInterval.sin,
                'cos': IntervalMatrixWithPsdInterval.cos,
                'exp': IntervalMatrixWithPsdInterval.exp,
                'ln': IntervalMatrixWithPsdInterval.ln,
                'log': IntervalMatrixWithPsdInterval.ln,
                'matrix_power': IntervalMatrixWithPsdInterval.matrix_power
            },
            'numpy'
        ]

    @staticmethod
    def _value_to_substitution(value):
        return IntervalMatrixWithPsdInterval.value_to_interval_matrix_with_psd_interval(value)

    @staticmethod
    def _default_substitution(shape=None):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix(shape=shape),
                                             psd_interval=PsdIntervalInformation(shape=shape))

    @staticmethod
    def _default_psd(shape=None):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix(shape=shape),
                                             psd_interval=PsdIntervalInformation(shape=shape, is_psd=True))

    @staticmethod
    def _identity(shape=None):
        return IntervalMatrixWithPsdInterval(matrix=IntervalMatrix.eye(shape[0]),
                                             psd_interval=PsdIntervalInformation(shape=shape, is_psd=True))
