import math
from abc import ABC, abstractmethod

import numpy as np


class EncDec(ABC):
    def __init__(self, max_dim, precision):
        self.max_dim = max_dim
        self.float_precision = precision
        self.symbols = ["V" + str(i) for i in range(1, self.max_dim + 1)]

    @abstractmethod
    def encode_value(self, val):
        pass

    @abstractmethod
    def decode_value(self, lst):
        pass

    def encode(self, matrix):
        n_row, n_col = matrix.shape
        encoding = ["V" + str(n_row), "V" + str(n_col)]
        for idx in range(n_row):
            for jdx in range(n_row):
                encoding.extend(self.encode_value(matrix[idx, jdx]))
        return encoding

    def decode(self, encoding):
        n_row = int(encoding[0].replace("V", ""))
        n_col = int(encoding[1].replace("V", ""))
        values = encoding[2:]
        matrix = np.zeros((n_row, n_col), dtype=float)
        for idx in range(n_row):
            for jdx in range(n_col):
                val = self.decode_value(values[idx * n_row + jdx])
                if np.isnan(val):
                    return None
                matrix[idx, jdx] = val
        return matrix


class FPSymbol(EncDec):
    def __init__(self, max_dim, precision, max_exp):
        super().__init__(max_dim, precision)
        self.max_exp = max_exp
        assert (self.float_precision + self.max_exp) % 2 == 0
        self.symbols.extend(["NaN", "-NaN"])
        dig = 10**self.float_precision
        self.logrange = (self.float_precision + self.max_exp) // 2
        self.base = 10 ** (self.logrange - self.float_precision)
        self.limit = 10**self.logrange
        self.symbols.extend(["N" + str(i) + "e0" for i in range(-dig + 1, dig)])
        for i in range(self.max_exp):
            for j in range(1, 10):
                for k in range(dig):
                    self.symbols.append("N" + str(j * dig + k) + "e" + str(i))
                    self.symbols.append("N-" + str(j * dig + k) + "e" + str(i))

    def encode_value(self, value):
        if abs(value) > self.limit:
            return ["NaN"] if value > 0 else ["-NaN"]
        sign = -1 if value < 0 else 1
        val_rebase = abs(value) * self.base
        if val_rebase == 0:
            return ["N0e0"]
        exp = int(math.log10(val_rebase))
        if exp < 0:
            exp = 0
        matissa = int(val_rebase * (10 ** (self.float_precision - exp)) + 0.5)
        if matissa == 0:
            sign = 1
        if (matissa == 1000) and (exp + 1 < self.max_exp):
            matissa = 100
            exp += 1
        pref = "N" if sign == 1 else "N-"
        return [pref + str(matissa) + "e" + str(exp)]

    def decode_value(self, value):
        if value == "NaN":
            return self.limit
        elif value == "-NaN":
            return -self.limit
        else:
            matissa, exp = value[1:].split("e")
            val = (int(matissa) * (10 ** int(exp))) / self.limit
            return val
