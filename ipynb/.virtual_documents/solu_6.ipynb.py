import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# seaborn is a package for the visualization of statistical data
import seaborn as sns
sns.set(style='ticks')


x = np.arange(1,11,1)
x

print('The mean of x:', x.mean() , 'and the SD of x:', np.std(x, ddof=1))



