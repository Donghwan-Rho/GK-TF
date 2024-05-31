from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import chebval
import numpy as np
import torch

coefs = [1, 2, 3]  # T0 + 2*T1 + 3*T2
p = Chebyshev(coefs)

# Numpy 다항식의 계수를 얻음
numpy_coefs = np.array(p.coef)

poly = p.convert(kind=Polynomial) # [-2, 2, 6] -> -2 + 2x + 6x^2

print(f'Chebyshev coeff: {p.coef}')
print("The coefficients of the polynomial in standard form are:",
      poly.coef)

x = 2
value = chebval(x, coefs)
print(f'value: {value}')

# Numpy 배열을 PyTorch 텐서로 변환
torch_coefs = torch.tensor(numpy_coefs, dtype=torch.float32)
print(f'torch_coefs: {torch_coefs}')

# PyTorch에서의 x 값을 정의 (예: x=2.0)
x = torch.tensor(2.0)

# Chebyshev 다항식 계산을 위한 x의 거듭제곱 준비
x_powers = torch.tensor([x**i for i in range(len(torch_coefs))])
print(f'x_powers: {x_powers}')

# 다항식 계산
poly_value = torch.dot(torch_coefs, x_powers)

print(f"The polynomial value at x={x.item()} is: {poly_value.item()}")