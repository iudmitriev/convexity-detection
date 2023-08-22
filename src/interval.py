from decimal import *


def decsig(value):
    """
    Вычисление знака числа как объекта класса Decimal
            Параметры:
                    value (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    sgn (Decimal): число, объект класса Decimal, знак исходного числа (Decimal("1") или Decimal("-1"))
    """
    return Decimal('1').copy_sign(Decimal(value))


def decisint(value):
    """
    Проверка объекта Decimal на принадлежность к целым числам
            Параметры:
                    value (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    result (bool): булево значение, где True - передано целое число
    """
    return (value // Decimal('1')) == value


def deciseven(value):
    """
    Проверка объекта Decimal на принадлежность к чётным числам
            Параметры:
                    value (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    result (bool): булево значение, где True - передано чётное целое число
    """
    return decisint(value) and (value % Decimal('2') == Decimal('0'))


def decisodd(value):
    """
    Проверка объекта Decimal на принадлежность к нечётным числам
            Параметры:
                    value (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    result (bool): булево значение, где True - передано нечётное целое число
    """
    return decisint(value) and (value % Decimal('2') == Decimal('1'))


def quantizestring(cnt):
    """
    Получение строки для метода Decimal.quantize для округления с нужной точностью
            Параметры:
                    cnt (int): число, необходимая точность округления
            Возвращаемое значение:
                    strquant (str): строка, конвертируемая в объект класса Decimal, для метода Decimal.quantize.
    """
    strquant = '0'
    if cnt <= 0:
        return '1'
    strquant = strquant + '.'
    for i in range(cnt - 1):
        strquant = strquant + '0'
    strquant = strquant + '1'
    return strquant


def decpi():
    """
    Вычисление константы - числа Пи как объекта класса Decimal
            Параметры:
                    -
            Возвращаемое значение:
                    s (Decimal): число, объект класса Decimal, значение числа Пи, вычисленное итеративным методом.
                        Точность: зависит от внешнего контекста.
                        Округление: зависит от внешнего контекста.
    """
    curcontext = getcontext().copy()
    getcontext().prec += 10
    getcontext().rounding = ROUND_HALF_EVEN
    three = Decimal("3")
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    setcontext(curcontext)
    return +s


def decsin(x):
    """
    Вычисление синуса числа - объекта класса Decimal
            Параметры:
                    x (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    s (Decimal): число, объект класса Decimal, значение sin(x), вычисленное
                            с помощью разложения в ряд Тейлора
                        Точность: зависит от внешнего контекста.
                        Округление: зависит от внешнего контекста.
    """
    curcontext = getcontext().copy()
    getcontext().prec += 10
    getcontext().rounding = ROUND_HALF_EVEN
    while x > 0:
        x -= Decimal("2") * decpi()
    while x < 0:
        x += Decimal("2") * decpi()
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign

    setcontext(curcontext)
    return +s


def deccos(x):
    """
    Вычисление косинуса числа - объекта класса Decimal
            Параметры:
                    x (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    s (Decimal): число, объект класса Decimal, значение cos(x),
                            вычисленное с помощью разложения в ряд Тейлора
                        Точность: зависит от внешнего контекста.
                        Округление: зависит от внешнего контекста.
    """
    curcontext = getcontext().copy()
    getcontext().prec += 10
    getcontext().rounding = ROUND_HALF_EVEN
    while x > 0:
        x -= Decimal("2") * decpi()
    while x < 0:
        x += Decimal("2") * decpi()
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    setcontext(curcontext)
    return +s


def dectg(x):
    """
    Вычисление тангенса числа - объекта класса Decimal
            Параметры:
                    x (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    s (Decimal): число, объект класса Decimal, значение tg(x) = sin(x) / cos(x);
                        Точность: зависит от внешнего контекста.
                        Округление: зависит от внешнего контекста.
    """
    getcontext().prec += 10
    s = decsin(x) / deccos(x)
    getcontext().prec -= 10
    return +s


def decctg(x):
    """
    Вычисление котангенса числа - объекта класса Decimal
            Параметры:
                    x (Decimal): число, объект класса Decimal
            Возвращаемое значение:
                    s (Decimal): число, объект класса Decimal, значение ctg(x) = cos(x) / sin(x);
                        Точность: зависит от внешнего контекста.
                        Округление: зависит от внешнего контекста.
    """
    getcontext().prec += 10
    s = deccos(x) / dectg(x)
    getcontext().prec -= 10
    return +s


class Interval:
    """
    Класс Interval - интервальная арифметика с управляемой точностью
        Поля:
            precision (int): точность результата, необходимое количество значащих цифр после запятой       | Default: 10
            calcprecision (int): точность вычислений, общее максимальное количество значащих цифр в числе  | Default: 50
            multiintervalmode (int): 0 - выключить результат деления из двух интервалов, 1 - включить      | Default: 1
        Вспомогательные методы взаимодействия с полями:
            void intervaldiv (): результат деления - всегда один интервал (multiintervalmode = 0)
            void multiintervaldiv (): результат деления может быть двумя интервалами (multiintervalmode = 1)
            void setprecision (int prec): установить количество значащих цифр после запятой в prec (precision = prec)
            void setcalcprecision (int prec): установить точность вычислений в x (calcprecision = prec)
        Операторы:
            +:  Interval __add__ (self, Interval): сумма двух интервалов с внешним расширяющим округлением
            -:  Interval __sub__ (self, Interval): разность двух интервалов с внешним расширяющим округлением
            *:  Interval __mul__ (self, Interval): произведение двух интервалов с внешним расширяющим округлением
            **: Interval __pow__ (self, Interval): возведение интервала в рациональную (только положительные интервалы)
                                                   или целую степень с внешним расширяющим округлением
            /:  Interval | [Interval, Interval] __truediv__ (self, Interval): частное двух интервалов с внешним
                                                                              расширяющим округлением, при
                                                                              multiintervalmode = 1 возможен возврат
                                                                              списка из двух объектов класса Interval
            <: Boolean __lt__ (self, Interval):   |
            <=: Boolean __le__ (self, Interval):  | операторы сравнения
            >: Boolean __gt__ (self, Interval):   | (true только при непересекающихся интервалах)
            >=: Boolean __ge__ (self, Interval):  |
            ==: Boolean __eq__ (self, Interval): оператор равенства
            !=: Boolean __ne__ (self, Interval): оператор неравенства

        Методы:
            Interval mid (self): возвращает точечный интервал - середину исходного интервала, с обычным округлением
            Interval scale (self, int factor): возвращает интервал с тем же центром, расширенный
                                         в factor раз, с внешним расширяющим округлением
            Decimal width (self): возвращает ширину интервала (точно)
            boolean isIn(self, Interval other): возвращает True, если исходный интервал внутри other, False иначе
            boolean isAround(self, Interval other): возвращает True, если other внутри исходного интервала, False иначе
            boolean | None sign(self): возвращает True, если интервал больше либо равен нулю, False, ecли интервал
                                       меньше либо равен нулю, None иначе
        Математические функции:
            Interval exp (self): интервал экспоненты данного интервала, с внешним расширяющим округлением
            Interval sin (self): интервал синуса данного интервала, с внешним расширяющим округлением
            Interval cos (self): интервал косинуса данного интервала, с внешним расширяющим округлением

    """
    precision = 10
    multiintervalmode = 1
    calcprecision = 50

    __savedcontext = Context(prec=50, Emax=MAX_EMAX, Emin=MIN_EMIN, traps=[])

    @staticmethod
    def __savecontext():
        Interval.__savedcontext = getcontext().copy()
        setcontext(Context(prec=Interval.calcprecision, Emax=MAX_EMAX, Emin=MIN_EMIN, traps=[]))

    @staticmethod
    def __loadcontext():
        setcontext(Interval.__savedcontext)

    def __init__(self, x):
        """
        Инициализация интервала
                Параметры:
                        x (List [x1, x2 ...]): список из хотя бы двух объектов, из которых можно создать
                                               объект класса Decimal
        """
        Interval.__savecontext()
        self.x = [0, 0]
        self.x[0] = Decimal(x[0])
        self.x[1] = Decimal(x[1])

        self.T = self
        self.__correctize()
        Interval.__loadcontext()

    def __repr__(self):
        Interval.__savecontext()
        self.__correctize()
        Interval.__loadcontext()
        return "[" + str(self.x[0]) + ", " + str(self.x[1]) + "]"

    def mid(self):
        """
        Получение середины интервала
                Возвращаемое значение:
                        middle (Decimal): число, объект класса Decimal, середина текущего интервала;
                            Точность: зависит от параметров Interval.
                            Округление: математическое.
        """
        Interval.__savecontext()
        middle = Decimal("Inf")
        if self.x[0] != Decimal("-Inf") and self.x[1] != Decimal("Inf"):
            middle = (Decimal("0.5") * (self.x[0] + self.x[1])).quantize(Decimal(quantizestring(self.precision)),
                                                                         rounding=ROUND_HALF_EVEN)
        Interval.__loadcontext()
        return middle

    def width(self):
        """
        Получение ширины интервала
                Возвращаемое значение:
                        width (Decimal): число, объект класса Decimal, ширина текущего интервала;
                            Точность: зависит от параметров Interval.
                            Округление: нет (всегда точное значение в текущих параметрах).
        """
        return self.x[1] - self.x[0]

    def scale(self, factor):
        """
        Расширение/сужение интервала при неизменном центре
                Параметры:
                        factor (...): число, объект, из которого можно создать объект класса Decimal
                Результат:
                        self расширяется в factor раз
        """
        Interval.__savecontext()
        m = Decimal("0.5") * (self.x[0] + self.x[1])
        r = Decimal("0.5") * (self.x[1] - self.x[0])
        getcontext().rounding = ROUND_FLOOR
        self.x[0] = (m - Decimal(factor) * r)
        getcontext().rounding = ROUND_CEILING
        self.x[1] = (m + Decimal(factor) * r)
        self.__correctize()
        Interval.__loadcontext()

    def isIn(self, other):
        """
        Проверка вложенности в другой интервал
                Параметры:
                        other (...): объект, из которого можно создать объект класса Interval;
                Возвращаемое значение:
                        result (boolean): результат предиката вложенности текущего интервала в другой интервал;
        """
        ointerval = Interval.valueToInterval(other)
        return (self.x[0] >= ointerval.x[0]) and (self.x[1] <= ointerval.x[1])

    def isAround(self, other):
        """
        Проверка вложенности другого интервала
                Параметры:
                        other (...): объект, из которого можно создать объект класса Interval;
                Возвращаемое значение:
                        result (boolean): результат предиката вложенности другого интервала в текущий интервал;
        """
        ointerval = Interval.valueToInterval(other)
        return (self.x[0] <= ointerval.x[0]) and (self.x[1] >= ointerval.x[1])

    def sign(self):
        """
                Знак интервала: True, если интервал больше либо равен нулю, False, ecли интервал меньше либо равен нулю,
                                None иначе
                        Возвращаемое значение:
                                result (boolean): результат предиката вложенности другого интервала в текущий интервал;
                """
        is_convex = self.isIn(Interval([0, float('inf')]))
        if is_convex:
            return True
        is_concave = self.isIn(Interval([float('-inf'), 0]))
        if is_concave:
            return False
        return None

    def __correctize(self):
        self.x = sorted(self.x)
        if self.x[0] != Decimal("Inf") and self.x[0] != Decimal("-Inf"):
            self.x[0] = self.x[0].quantize(Decimal(quantizestring(self.precision)), rounding=ROUND_FLOOR)
        if self.x[1] != Decimal("Inf") and self.x[1] != Decimal("-Inf"):
            self.x[1] = self.x[1].quantize(Decimal(quantizestring(self.precision)), rounding=ROUND_CEILING)

    def __getitem__(self, item):
        Interval.__savecontext()
        self.__correctize()
        Interval.__loadcontext()
        return self.x[item]

    def __setitem__(self, key, value):
        Interval.__savecontext()
        self.x.__setitem__(key, Decimal(value))
        self.__correctize()
        Interval.__loadcontext()

    def __abs__(self):
        if self > 0:
            return Interval(self.x.copy())
        if self < 0:
            return -Interval(self.x.copy())
        return Interval([0, max(-self[0], self[1])])

    def __neg__(self):
        ninterval = Interval(self.x)
        Interval.__savecontext()
        ninterval.x[0] = - self.x[1]
        ninterval.x[1] = - self.x[0]
        ninterval.__correctize()
        Interval.__loadcontext()
        return ninterval

    def __add__(self, other):
        ointerval = Interval.valueToInterval(other)
        ninterval = Interval(self.x)
        Interval.__savecontext()
        getcontext().rounding = ROUND_FLOOR
        ninterval.x[0] = self.x[0] + ointerval.x[0]
        getcontext().rounding = ROUND_CEILING
        ninterval.x[1] = self.x[1] + ointerval.x[1]
        ninterval.__correctize()
        Interval.__loadcontext()
        return ninterval

    def __pow__(self, other):
        Interval.__savecontext()
        ointerval = Interval.valueToInterval(other)
        if (not decisint(ointerval.x[0])) or (not decisint(ointerval.x[1])):
            if self.x[0] < Decimal("0") or self.x[1] < Decimal("0"):
                Interval.__loadcontext()
                return Interval(["NAN", "NAN"])
            else:
                getcontext().rounding = ROUND_FLOOR
                vrd = [self.x[0] ** ointerval.x[0], self.x[0] ** ointerval.x[1],
                       self.x[1] ** ointerval.x[0], self.x[1] ** ointerval.x[1]]
                getcontext().rounding = ROUND_CEILING
                vru = [self.x[0] ** ointerval.x[0], self.x[0] ** ointerval.x[1],
                       self.x[1] ** ointerval.x[0], self.x[1] ** ointerval.x[1]]
                ninterval = Interval([min(vrd), max(vru)])
                Interval.__loadcontext()
                return ninterval
        else:
            if ointerval.x[0] == ointerval.x[1]:
                if deciseven(ointerval.x[0]) and decsig(self.x[0] != decsig(self.x[1])):
                    getcontext().rounding = ROUND_CEILING
                    vru = [self.x[0] ** ointerval.x[0], self.x[1] ** ointerval.x[0]]
                    ninterval = Interval([0, max(vru)])
                    Interval.__loadcontext()
                    return ninterval
                getcontext().rounding = ROUND_FLOOR
                vrd = [self.x[0] ** ointerval.x[0], self.x[1] ** ointerval.x[0]]
                getcontext().rounding = ROUND_CEILING
                vru = [self.x[0] ** ointerval.x[0], self.x[1] ** ointerval.x[0]]
                ninterval = Interval([min(vrd), max(vru)])
                Interval.__loadcontext()
                return ninterval
            else:
                getcontext().rounding = ROUND_FLOOR
                vrd = [self.x[0] ** ointerval.x[0], self.x[0] ** ointerval.x[1],
                       self.x[1] ** ointerval.x[0], self.x[1] ** ointerval.x[1],
                       self.x[0] ** (ointerval.x[0] + 1), self.x[0] ** (ointerval.x[1] - 1),
                       self.x[1] ** (ointerval.x[0] + 1), self.x[1] ** (ointerval.x[1] - 1)]
                getcontext().rounding = ROUND_CEILING
                vru = [self.x[0] ** ointerval.x[0], self.x[0] ** ointerval.x[1],
                       self.x[1] ** ointerval.x[0], self.x[1] ** ointerval.x[1],
                       self.x[0] ** (ointerval.x[0] + 1), self.x[0] ** (ointerval.x[1] - 1),
                       self.x[1] ** (ointerval.x[0] + 1), self.x[1] ** (ointerval.x[1] - 1)]
                ninterval = Interval([min(vrd), max(vru)])
                Interval.__loadcontext()
                return ninterval

    def __radd__(self, other):
        ointerval = Interval.valueToInterval(other)
        return ointerval.__add__(self)

    def __sub__(self, other):
        ointerval = Interval.valueToInterval(other)
        ninterval = Interval(self.x)
        Interval.__savecontext()
        getcontext().rounding = ROUND_FLOOR
        ninterval.x[0] = self.x[0] - ointerval.x[1]
        getcontext().rounding = ROUND_CEILING
        ninterval.x[1] = self.x[1] - ointerval.x[0]
        ninterval.__correctize()
        Interval.__loadcontext()
        return ninterval

    def __rsub__(self, other):
        ointerval = Interval.valueToInterval(other)
        return ointerval.__sub__(self)

    def __mul__(self, other):
        ointerval = Interval.valueToInterval(other)
        Interval.__savecontext()
        getcontext().rounding = ROUND_FLOOR
        vrd = [self.x[0] * ointerval.x[0], self.x[0] * ointerval.x[1],
               self.x[1] * ointerval.x[0], self.x[1] * ointerval.x[1]]
        getcontext().rounding = ROUND_CEILING
        vru = [self.x[0] * ointerval.x[0], self.x[0] * ointerval.x[1],
               self.x[1] * ointerval.x[0], self.x[1] * ointerval.x[1]]
        b = [min(vrd), max(vru)]
        Interval.__loadcontext()
        return Interval(b)  # __correctize inside

    def __rmul__(self, other):
        ointerval = Interval.valueToInterval(other)
        return ointerval.__mul__(self)

    def __getNullType(self):
        """
            0: -0 or +0 not present
            1: -0 present
            2: +0 present
            3: -0 and +0 present
            4: NaN present
        """
        if Decimal.is_nan(self.x[0]) or Decimal.is_nan(self.x[1]):
            return 4
        if not self.isAround('0'):
            return 0
        if decsig(self.x[0]) != decsig(self.x[1]):
            return 3
        if decsig(self.x[0]) == 1:
            return 2
        if decsig(self.x[0]) == -1:
            return 1
        return 4


    def __truediv__(self, other):
        ointerval = Interval.valueToInterval(other)
        Interval.__savecontext()
        stype = self.__getNullType()
        otype = ointerval.__getNullType()
        if stype == 4 or otype == 4:
            Interval.__loadcontext()
            return Interval(["nan", "nan"])

        if (stype == 3 and otype == 0) or (otype < 3 and stype < 3):
            getcontext().rounding = ROUND_FLOOR
            vrd = [self.x[0] / ointerval.x[0], self.x[0] / ointerval.x[1],
                   self.x[1] / ointerval.x[0], self.x[1] / ointerval.x[1]]
            getcontext().rounding = ROUND_CEILING
            vru = [self.x[0] / ointerval.x[0], self.x[0] / ointerval.x[1],
                   self.x[1] / ointerval.x[0], self.x[1] / ointerval.x[1]]
            vrd = [i for i in vrd if (not Decimal.is_nan(i))]
            vru = [i for i in vru if (not Decimal.is_nan(i))]
            if len(vrd) == 0 or len(vru) == 0:
                if stype == otype:
                    Interval.__loadcontext()
                    return Interval(["0", "Inf"])
                else:
                    Interval.__loadcontext()
                    return Interval(["-Inf", "-0"])
            b = [min(vrd), max(vru)]
            return Interval(b)

        if otype == 3 and stype == 0:
            if not Interval.multiintervalmode:
                Interval.__loadcontext()
                return Interval(["-Inf", "Inf"])
            b1 = self / Interval([ointerval.x[0], Decimal("-0")])
            b2 = self / Interval([Decimal("0"), ointerval.x[1]])
            Interval.__loadcontext()
            return [b1, b2]

        Interval.__loadcontext()
        return Interval(["-Inf", "Inf"])

    def __lt__(self, other):
        ointerval = Interval.valueToInterval(other)
        return self.x[1] < ointerval.x[0]

    def __le__(self, other):
        ointerval = Interval.valueToInterval(other)
        return self.x[1] <= ointerval.x[0]

    def __gt__(self, other):
        ointerval = Interval.valueToInterval(other)
        return self.x[0] > ointerval.x[1]

    def __ge__(self, other):
        ointerval = Interval.valueToInterval(other)
        return self.x[0] >= ointerval.x[1]

    def __eq__(self, other):
        ointerval = Interval.valueToInterval(other)
        return self.x[0] == ointerval.x[0] and self.x[1] == ointerval.x[1]

    def __ne__(self, other):
        return not self.__eq__(other)


    @staticmethod
    def intervaldiv():
        """
        Переключение в режим одноинтервального деления
                Результат:
                        Любая операция деления (__truediv__) объектов Interval будет возвращать только один интервал;
        """
        Interval.multiintervalmode = 0


    @staticmethod
    def multiintervaldiv():
        """
        Переключение в режим мультиинтервального деления
                Результат:
                        Операция деления (__truediv__) объектов Interval сможет возвращать список из двух интервалов;
        """
        Interval.multiintervalmode = 1


    @staticmethod
    def setprecision(prec):
        """
        Установка точности результата
                Параметры:
                        prec (int): число, новая точность результатов операций Interval;
                Результат:
                        Точность результатов операций Interval становится равной prec;
        """
        Interval.precision = prec


    @staticmethod
    def setcalcprecision(prec):
        """
        Установка точности вычислений
                Параметры:
                        prec (int): число, новая точность вычислений операций Interval;
                Результат:
                        Точность вычислений операций Interval становится равной prec;
        """
        Interval.calcprecision = prec


    @staticmethod
    def valueToInterval(expr):
        """
        Создание объекта Interval из int, float, str, Decimal и списка двух объектов
                Параметры:
                        1. expr (...): объект, из которого можно создать объект класса Decimal;
                        2. expr ([x1, x2..]): список из хотя бы двух элементов, из которых можно создать объекты
                                              класса Decimal;
                Возвращаемое значение:
                        result (Interval): созданный объект класса Interval;
                            Точность: зависит от внешнего контекста.
                            Округление: зависит от внешнего контекста.
        """
        if isinstance(expr, int):
            etmp = Interval([expr, expr])
        elif isinstance(expr, float):
            etmp = Interval([expr, expr])
        elif isinstance(expr, str):
            etmp = Interval([expr, expr])
        elif isinstance(expr, Decimal):
            etmp = Interval([expr, expr])
        else:
            etmp = expr
        return etmp


    @staticmethod
    def sin(x):
        """
        Вычисление интервала - синуса интервала
                Параметры:
                        x (Interval): объект класса Interval;
                Возвращаемое значение:
                        result (Interval): объект класса Interval, синус исходного интервала;
                            Точность: зависит от параметров Interval.
                            Округление: внешнее расширяющее.
        """
        Interval.__savecontext()
        ed = Decimal(quantizestring(Interval.precision)).quantize(Decimal(quantizestring(Interval.precision)),
                                                                  rounding=ROUND_CEILING)
        getcontext().rounding = ROUND_FLOOR
        yrd = [decsin(x[0]) - ed, decsin(x[1]) - ed]
        getcontext().rounding = ROUND_CEILING
        yru = [decsin(x[0]) + ed, decsin(x[1]) + ed]
        pi2 = Decimal("2") * decpi()
        pi05 = decpi() / Decimal("2")
        if ((x[0] - pi05) / pi2).quantize(Decimal("1"), rounding=ROUND_CEILING) <= ((x[1] - pi05) / pi2).quantize(
                Decimal("1"), rounding=ROUND_FLOOR):
            b = Decimal("1")
        else:
            b = max(yru)
        if ((x[0] + pi05) / pi2).quantize(Decimal("1"), rounding=ROUND_CEILING) <= ((x[1] + pi05) / pi2).quantize(
                Decimal("1"), rounding=ROUND_FLOOR):
            a = Decimal("-1")
        else:
            a = min(yrd)
        Interval.__loadcontext()
        return Interval([a, b])


    @staticmethod
    def cos(x):
        """
        Вычисление интервала - косинуса интервала
                Параметры:
                        x (Interval): объект класса Interval;
                Возвращаемое значение:
                        result (Interval): объект класса Interval, косинус исходного интервала;
                            Точность: зависит от параметров Interval.
                            Округление: внешнее расширяющее.
        """
        Interval.__savecontext()
        ed = Decimal(quantizestring(Interval.precision)).quantize(Decimal(quantizestring(Interval.precision)),
                                                                  rounding=ROUND_CEILING)
        getcontext().rounding = ROUND_FLOOR
        yrd = [deccos(x[0]) - ed, deccos(x[1]) - ed]
        getcontext().rounding = ROUND_CEILING
        yru = [deccos(x[0]) + ed, deccos(x[1]) + ed]
        pi2 = 2 * decpi()
        if (x[0] / pi2).quantize(Decimal("1"), rounding=ROUND_CEILING) <= (x[1] / pi2).quantize(Decimal("1"),
                                                                                                rounding=ROUND_FLOOR):
            b = Decimal("1")
        else:
            b = max(yru)
        if ((x[0] - decpi()) / pi2).quantize(Decimal("1"), rounding=ROUND_CEILING) <= ((x[1] - decpi()) / pi2).quantize(
                Decimal("1"), rounding=ROUND_FLOOR):
            a = Decimal("-1")
        else:
            a = min(yrd)
        Interval.__loadcontext()
        return Interval([a, b])


    @staticmethod
    def exp(x):
        """
        Эксопонента интервала
                Параметры:
                        x (...): объект, из которого можно создать объект класса Interval;
                Возвращаемое значение:
                        result (Interval): новый интервал, соответствующий экспоненте от исходного;
        """
        ninterval = Interval(x)
        Interval.__savecontext()
        getcontext().rounding = ROUND_FLOOR
        ninterval.x[0] = Decimal.exp(ninterval.x[0])
        getcontext().rounding = ROUND_CEILING
        ninterval.x[1] = Decimal.exp(ninterval.x[1])
        ninterval.__correctize()
        Interval.__loadcontext()
        return ninterval


    @staticmethod
    def ln(x):
        """
        Натуральный логарифм интервала
                Параметры:
                        x (...): объект, из которого можно создать объект класса Interval;
                Возвращаемое значение:
                        result (Interval): новый интервал, соответствующий натуралному логарифму от исходного;
        """
        ninterval = Interval(x)
        Interval.__savecontext()
        getcontext().rounding = ROUND_FLOOR
        ninterval.x[0] = Decimal.ln(ninterval.x[0])
        getcontext().rounding = ROUND_CEILING
        ninterval.x[1] = Decimal.ln(ninterval.x[1])
        ninterval.__correctize()
        Interval.__loadcontext()
        return ninterval
