from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    num1: float
    num2: float

    def add(self):
        return self.num1 + self.num2

    def subtract(self):
        return self.num1 - self.num2

    def multiply(self):
        return self.num1 * self.num2

    def divide(self):
        if self.num2 != 0:
            return self.num1 / self.num2
        else:
            raise ValueError('Do not divide by zero!')
