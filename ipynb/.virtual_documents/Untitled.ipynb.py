import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats

# seaborn is a package for the visualization of statistical data
import seaborn as sns
sns.set(style='ticks')


df = pd.read_csv('C:\\Users\\magda\\Desktop\\ind2019er\\pi.csv')
x = df.ER30012


df.ER30012.mean()

np.std(df.ER30012, ddof=1)

df.ER30012.describe()


rv = stats.norm(360,1510)
x2 = np.r_[0:1:0.001]
print(x2.size)


df.ER30012.hist( cumulative = True,density=1, bins=10)


stats_df = df.groupby('ER30012')['ER30012'].agg('count').pipe(pd.DataFrame).rename(columns = {'ER30012': 'frequency'})

# PDF
stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

# CDF
stats_df['cdf'] = stats_df['pdf'].cumsum()
stats_df = stats_df.reset_index()
stats_df.drop([0], inplace=True)
stats_df


# stats_df.plot(x = 'ER30012', y = ['pdf', 'cdf'], grid = True)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(stats_df.ER30012, stats_df.pdf, 'g-')
ax2.plot(stats_df.ER30012, stats_df.cdf, 'b-')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')

plt.show()
