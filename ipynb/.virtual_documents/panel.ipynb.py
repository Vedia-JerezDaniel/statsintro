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


df.ER30012.hist(density=1, bins=10)


stats_df = df.groupby('ER30012')['ER30012'].agg('count').pipe(pd.DataFrame).rename(columns = {'ER30012': 'frequency'})

# PDF
stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

# CDF
stats_df['cdf'] = stats_df['pdf'].cumsum()
stats_df['sft'] = 1 - stats_df['cdf']
stats_df = stats_df.reset_index()
stats_df.drop([0], inplace=True)
stats_df


stats_df.ER30012.describe()


# stats_df.plot(x = 'ER30012', y = ['pdf', 'cdf'], grid = True)
plt.figure(figsize=(12,12))
ax = plt.subplot2grid((2,2),(0,0), colspan=2)
stats_df.ER30012.hist(density=5, bins=100, color='green')
plt.axvline(x=3520, color='black', ls='--', lw=2)
# plt.plot(stats_df.ER30012, stats_df.pdf,  'gs')
plt.title('Normal Distribution - PDF')

plt.subplot(223)
plt.plot(stats_df.ER30012, stats_df.cdf, 'b-')
plt.axvline(x=3520, color='r')
plt.title('CDF: cumulative distribution fct')

plt.subplot(224)
plt.plot(stats_df.ER30012, stats_df.sft, 'b-')
plt.axvline(x=3520, color='r')
plt.title('SF: survival fct')

plt.show()
