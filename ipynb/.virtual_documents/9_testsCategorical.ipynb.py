import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Get the data
numTotal = 215
numPositive = 39

# Calculate the confidence intervals
p = float(numPositive)/numTotal
se = np.sqrt(p*(1-p)/numTotal)
td = stats.t(numTotal-1)
ci = p + np.array([-1,1])*td.isf(0.025)*se

# Print them
print('ONE PROPORTION')
print(p)
print('The confidence interval for the given sample is {0:5.3f} to {1:5.3f}'.format(
    ci[0], ci[1]))


# Enter the data
obs = np.array([[32, 118, 178, 299], [17, 127, 189, 292]])

# Calculate the chi-square test
chi2_corrected = stats.chi2_contingency(obs, correction=True)
chi2_uncorrected = stats.chi2_contingency(obs, correction=False)

# Print the result
print('CHI SQUARE')
print('The corrected chi2 value is {0:5.3f}, with p={1:5.3f}'.format(chi2_corrected[0], chi2_corrected[1]))
print('The uncorrected chi2 value is {0:5.3f}, with p={1:5.3f}'.format(chi2_uncorrected[0], chi2_uncorrected[1]))


# Enter the data
obs = np.array([[1, 23], [8, 18]])

# Calculate the Fisher Exact Test
fisher_result = stats.fisher_exact(obs)

# Print the result
print('The probability of obtaining a distribution at least as extreme '
+ 'as the one that was actually observed, assuming that the null ' +
    'hypothesis is true, is: {0:5.3f}.'.format(fisher_result[1]))


from statsmodels.sandbox.stats.runs import cochrans_q
import pandas as pd

tasks = np.array([[0,1,1,0,1,0,0,1,0,0,0,0,1,1,0,1],
                  [1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1],
                  [0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0]])

# I prefer a DataFrame here, as it indicates directly what the values mean
df = pd.DataFrame(tasks.T, columns = ['Task1', 'Task2', 'Task3'])

# --- >>> START stats <<< ---
(Q, pVal) = cochrans_q(df)
# --- >>> STOP stats <<< ---

print('Q = {0:5.3f}, p = {1:5.3f}'.format(Q, pVal))
if pVal < 0.05:
    print("There is a significant difference between the three tasks.")



from statsmodels.sandbox.stats.runs import mcnemar

f_obs = np.array([[101, 121],[59, 33]])
(statistic, pVal) = mcnemar(f_obs)

print('p = {0:5.3e}'.format(pVal))
if pVal < 0.05:
    print("There was a significant change in the disease by the treatment.") 
