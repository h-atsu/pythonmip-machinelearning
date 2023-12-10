from abc import ABC, abstractmethod
import numpy as np


class AbstractFunction(ABC):

    @abstractmethod
    def __call__(self, x):
        """
        Apply the function to the input x.

        :param x: The input to the function.
        :return: The result of applying the function to x.
        """
        pass

    @abstractmethod
    def df(self, x):
        """
        Compute the derivative of the function at x.

        :param x: The point at which to compute the derivative.
        :return: The derivative of the function at x.
        """
        pass

    @abstractmethod
    def df_inv(self, g, x):
        """
        Compute the inverse derivative of the function at x.

        :param g: The gradient at which to compute the inverse derivative.
        :param x: The position at which to compute the inverse derivative.
        :return: The inverse derivative of the function at x.
        """
        pass


class Log(AbstractFunction):
    def __init__(self, lb=1e-9, ub=1e6):
        self.lb = lb
        self.ub = ub
        self.infrection_pts = []

    def __call__(self, x):
        return np.log(x)

    def f(self, x):
        return np.log(x)

    def df(self, x):
        return 1/x

    def df_inv(self, g, x):
        return 1/g


class Exp(AbstractFunction):
    def __init__(self, lb=-1e6, ub=1e6):
        self.lb = lb
        self.ub = ub
        self.infrection_pts = []

    def __call__(self, x):
        return np.exp(x)

    def f(self, x):
        return np.exp(x)

    def df(self, x):
        return np.exp(x)

    def df_inv(self, g, x):
        return np.log(g)


class Logistic(AbstractFunction):
    def __init__(self, lb=-1e6, ub=1e6):
        self.lb = lb
        self.ub = ub
        self.infrection_pts = [0]

    def __call__(self, x):
        return 1/(1+np.exp(-x))

    def f(self, x):
        return 1/(1+np.exp(-x))

    def df(self, x):
        y = self.f(x)
        return y*(1-y)

    def df_inv(self, g, x):
        if x >= 0:
            return -np.log(2/(1+np.sqrt(1-4*g))-1)
        else:
            return -np.log(2/(1-np.sqrt(1-4*g))-1)


class Hill(AbstractFunction):
    def __init__(self, lb=0, ub=1e6, params={'a': 1, 'b': 1, 'c': 1}):
        self.lb = lb
        self.ub = ub
        self.infrection_pts = []
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        assert self.a > 0 and self.b > 0 and self.c > 0

    def __call__(self, x):
        return self.a*x / (self.b*x + self.c)

    def f(self, x):
        return self.a*x / (self.b*x + self.c)

    def df(self, x):
        return self.a*self.c / (self.b*x + self.c)**2

    def df_inv(self, g, x):
        return (np.sqrt(self.a * self.c / g) - self.c)/self.b
