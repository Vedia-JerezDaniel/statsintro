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


stats_df = df.groupby('ER30012')['ER30012'].agg('count').pipe(pd.DataFrame).rename(columns = {'ER30012': 'frequency'})

# PDF
stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

# CDF
stats_df['cdf'] = stats_df['pdf'].cumsum()
stats_df['sft'] = 1 - stats_df['cdf']
stats_df = stats_df.reset_index()
stats_df.drop([0], inplace=True)
stats_df

x =  stats_df.ER30012


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


stats.probplot(x, plot=plt)


# additional packages
from statsmodels.stats.diagnostic import lilliefors

def check_normality():
    '''Check if the distribution is normal.'''
    
    # # Set the parameters
    # numData = 1000
    # myMean = 0
    # mySD = 3
    
    # # To get reproducable values, I provide a seed value
    # np.random.seed(1234)   
    
    # Generate and show random data
    # data = stats.norm.rvs(myMean, mySD, size=numData)
    data = x
    fewData = data[:int(len(data)*.1)]
    plt.hist(data)
    plt.title("Data Histogram")
    plt.show()

    # --- >>> START stats <<< ---
    # Graphical test: if the data lie on a line, they are pretty much
    # normally distributed
    _ = stats.probplot(data, plot=plt)
    plt.show()

    pVals = pd.Series()
    pFewVals = pd.Series()
    # The scipy normaltest is based on D-Agostino and Pearsons test that
    # combines skew and kurtosis to produce an omnibus test of normality.
    _, pVals['Omnibus']    = stats.normaltest(data)
    _, pFewVals['Omnibus'] = stats.normaltest(fewData)

    # Shapiro-Wilk test
    _, pVals['Shapiro-Wilk']    = stats.shapiro(data)
    _, pFewVals['Shapiro-Wilk'] = stats.shapiro(fewData)
    
    # Or you can check for normality with Lilliefors-test
    _, pVals['Lilliefors']    = lilliefors(data)
    _, pFewVals['Lilliefors'] = lilliefors(fewData)
    
    # Alternatively with original Kolmogorov-Smirnov test
    _, pVals['Kolmogorov-Smirnov']    = stats.kstest((data-np.mean(data))/np.std(data,ddof=1), 'norm')
    _, pFewVals['Kolmogorov-Smirnov'] = stats.kstest((fewData-np.mean(fewData))/np.std(fewData,ddof=1), 'norm')
    
    print('p-values for all {0} data points: ----------------'.format(len(data)))
    print(pVals)
    print('p-values for the first 10 percent data points: ----------------')
    print(pFewVals)
    
    if pVals['Omnibus'] > 0.05:
        print('Data are normally distributed')
    # --- >>> STOP stats <<< ---
    
    return pVals['Kolmogorov-Smirnov']
    
if __name__ == '__main__':
    p = check_normality()    
    print(p)



df.ER30080.replace(0, None, inplace=True)

plt.scatter(df.ER30080, df.ER30057)


stats.normaltest(df.ER30080)


stats.ttest_1samp(x, 35200)


stats.ttest_rel(x, df.ER30057)


def check_mean():
    '''Data from Altman, check for significance of mean value.
    Compare average daily energy intake (kJ) over 10 days of 11 healthy women, and compare it to the recommended level of 7725 kJ.
    '''
    # Get data from Altman
    # inFile = 'altman_91.txt'
    # data = np.genfromtxt(inFile, delimiter=',')

    # Watch out: by default the standard deviation in numpy is calculated with ddof=0, corresponding to 1/N!
    myMean = np.mean(x)
    mySD = np.std(x, ddof=1)     # sample standard deviation
    print(('Mean and SD: {0:4.2f} and {1:4.2f}'.format(myMean, mySD)))

    # Confidence intervals
    tf = stats.t(len(x)-1)
    # multiplication with np.array[-1,1] is a neat trick to implement "+/-"
    ci = np.mean(x) + stats.sem(x)*np.array([-1, 1])*tf.ppf(0.975)
    print(
        ('The confidence intervals are {0:4.2f} to {1:4.2f}.'.format(ci[0], ci[1])))

    # Check if there is a significant difference relative to "checkValue"
    checkValue = 325
    # --- >>> START stats <<< ---
    t, prob = stats.ttest_1samp(x, checkValue)
    if prob < 0.05:
        print(('{0:4.2f} is significantly different from the mean (p={1:5.3f}).'.format(
            checkValue, prob)))

    # For not normally distributed data, use the Wilcoxon signed rank sum test
    (rank, pVal) = stats.wilcoxon(x-checkValue)
    issignificant = 'unlikely' if pVal < 0.05 else 'likely'
    # --- >>> STOP stats <<< ---
    print(('It is ' + issignificant + ' that the value is {0:d}'.format(checkValue)))
    return prob  # should be 0.018137235176105802


def compareWithNormal():
    '''This function is supposed to give you an idea how big/small the difference between t- and normal
    distribution are for realistic calculations.
    '''
    # generate the data
    np.random.seed(12345)
    normDist = stats.norm(loc=320, scale=1520)
    data = normDist.rvs(100)
    checkVal = 6.5

    # T-test
    # --- >>> START stats <<< ---
    t, tProb = stats.ttest_1samp(data, checkVal)
    # --- >>> STOP stats <<< ---

    # Comparison with corresponding normal distribution
    mmean = np.mean(data)
    mstd = np.std(data, ddof=1)
    normProb = stats.norm.cdf(
        checkVal, loc=mmean, scale=mstd/np.sqrt(len(data)))*2
    # compare
    print(('The probability from the t-test is ' +
          '{0:5.4f}, and from the normal distribution {1:5.4f}'.format(tProb, normProb)))

    return normProb  # should be 0.054201154690070759



if __name__ == '__main__':
    check_mean()
    compareWithNormal()
