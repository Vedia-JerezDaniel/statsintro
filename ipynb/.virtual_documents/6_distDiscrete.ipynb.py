# Note: here I use the modular approach, which is more appropriate for scripts
# "get_ipython().run_line_magic("pylab", " inline\" also loads numpy as np, and matplotlib.pyplot as plt")

get_ipython().run_line_magic("pylab", " inline")
import scipy.stats as stats


bd1 = stats.binom(20, 0.5)
bd2 = stats.binom(20, 0.7)
bd3 = stats.binom(40, 0.5)
k = arange(40)

plot(k, bd1.pmf(k), 'o-b')
plot(k, bd2.pmf(k), 'd-r')
plot(k, bd3.pmf(k), 's-g')

title('Binomial distribition')
legend(['p=0.5 and n=20', 'p=0.7 and n=20', 'p=0.5 and n=40'])
xlabel('X')
ylabel('P(X)')


pd = stats.poisson(10)
plot(k, pd.pmf(k),'x-')
title('Poisson distribition - PMF')
xlabel('X')
ylabel('P(X)')


k = arange(30)
plot(k, pd.cdf(k))
title('Poisson distribition - CDF')
xlabel('X')
ylabel('P(X)')


y = linspace(0,1,100)
plot(y, pd.ppf(y))
title('Poisson distribition - PPF')
xlabel('X')
ylabel('P(X)')
