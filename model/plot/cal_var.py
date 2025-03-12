import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

data = pd.read_csv( 'multi_run_syn_00.csv')
# print( data[['cov_type', 'lambda', 'gamma', 'eff_mean', 'picp_mean']].groupby(['cov_type', 'lambda', 'gamma']).agg(['mean','var']) )

# print(data)
df = data[ (data['cov_type'] == 'ellip') & (data['lambda'] == 0)  & (data['gamma'] == 0)] ['picp_mean']
print(np.mean(df) )
# print(df)


mu_greater = 95
mu_less = 95
# Compute sample mean and standard deviation
sample_mean = df.mean()
sample_std = df.std(ddof=1)  # Unbiased standard deviation
n = len(df)

# One-sample t-tests
t_stat_greater, p_value_greater = stats.ttest_1samp(df, mu_greater)
t_stat_less, p_value_less = stats.ttest_1samp(df, mu_less)

# Convert to one-tailed p-values
p_value_greater_one_tailed = p_value_greater / 2 if t_stat_greater > 0 else 1 - (p_value_greater / 2)
p_value_less_one_tailed = p_value_less / 2 if t_stat_less < 0 else 1 - (p_value_less / 2)

# Significance level
alpha = 0.05

# Determine hypothesis test results
reject_greater = "Reject H0: Mean is significantly greater than 0.95" if p_value_greater_one_tailed < alpha else "Fail to reject H0: No significant evidence that mean > 0.95"
reject_less = "Reject H0: Mean is significantly less than 95" if p_value_less_one_tailed < alpha else "Fail to reject H0: No significant evidence that mean < 95"

# Print results
print(reject_greater, reject_less)