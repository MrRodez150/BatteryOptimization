
import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

limits = np.array([[0.2, 4.0],
                   [12e-6, 30e-6],
                   [40e-6, 250e-6],
                   [10e-6, 100e-6],
                   [40e-6, 150e-6],
                   [12e-6, 30e-6],
                   [40e-3, 100e-3],
                   [0.2e-6, 20e-6],
                   [0.5e-6, 50e-6],
                   [4e-3, 25e-3],
                   [0.3, 0.6],
                   [0.3, 0.6],
                   [0.3, 0.6]])
sampling = LHS(xlimits=limits)

num = 5
x = sampling(num)

print(x.shape)
print(x)
