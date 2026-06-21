import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import seaborn as sns
import os as os

from matplotlib import pyplot as plt

population = stats.norm(loc=4, scale=0.8)

def calc_sample_mean(size, n_trial):
  sample_mean_array = np.zeros(n_trial)
  for i in range(0, n_trial):
    sample = population.rvs(size=size)
    sample_mean_array[i] = np.mean(sample)
  return(sample_mean_array)

# base_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(base_dir, "sample/3-4-1-fish_length_100000.csv")

# sns.set_theme()

size_array = np.arange(start=2, stop=1002, step=2)
sample_mean_std_array = np.zeros(len(size_array))
np.random.seed(1)
for i in range(0, len(size_array)):
  sample_mean = calc_sample_mean(size=size_array[i], n_trial=100)
  sample_mean_std_array[i] = np.std(sample_mean, ddof=1)

plt.plot(size_array, sample_mean_std_array, color='black')
plt.xlabel('sample size')
plt.ylabel('mean_std value')

plt.show()