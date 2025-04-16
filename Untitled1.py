Practical - 1: Write a program to implement Problems based on binomial distribution
import math

def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def BD(n, p, r):
    q = 1 - p
    return nCr(n, r) * p**r * q**(n-r)

print("Binomial Distribution Formula:")
print("P(r) = nCr(n, r) * p**r * q**(n-r)")

n = int(input("Enter N (Number of trails) : "))
p = float(input("Enter P (Probality of success): "))
r = int(input("Enter r (Number of success in N trails): "))

print("Mean", n * p)
print("SD:", math.sqrt(n * p * p - 1))
print(round(BD(n, p, r) * 100, 3),"%Chance")

Practical - 2: Write a program to implement Problems based on normal distribution-------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(1, 50, 100)
print(x)
def normal_dist(x, mean, sd):
    prob_density = norm.pdf(x, mean, sd)
    return prob_density

def SND(value, mean, sd):
    return (value - mean) / sd

mean = np.mean(x)
sd = np.std(x)
print("Mean:", mean)
print("SD:", sd)

# probability density function (PDF) of a normal (Gaussian) distribution:
pdf = normal_dist(x, mean, sd)
print(pdf)
print("Standard Normal Distribution Formula: z = (value - mean) / sd")
print("Mean:", mean)
print("SD:", sd)
num = float(input("Enter the value:"))
z = round(SND(num, mean, sd), 3)

print("Therefore, z = ", z)

Practical - 3: Write a program to implement Property plotting of binomial distribution---------------------------------------------------------------------------------------------------------------
import numpy as np
import math
from scipy.stats import binom

x = np.arange(0, n+1)

pmf = binom.pmf(x, n, p)

plt.bar(x, pmf, color = "yellow", alpha = 0.7, edgecolor = "blue")
plt.xlabel("Number of Success")
plt.ylabel("Probability")
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.show()

Practical - 4: Write a program to implement Property plotting of normal distribution---------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.plot(x, pdf, color = "green")
plt.xlabel("Data points")
plt.ylabel("Probability Density")
plt.show()

values = np.random.normal(mean, sd, 100)
plt.hist(values, 200, alpha=0.6, color='blue', edgecolor='black')
plt.show()

Practical - 5: Write a program to implement Plotting pdf, cdf, pmf, for discrete and continuous distribution-------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# For discrete distribution
n = int(input("Enter a Number of trails:"))
p = float(input("Enter a probabolity of success:"))
k = np.arange(0, n+1)

pmf_values = binom.pmf(k, n, p)
cdf_values = binom.cdf(k, n,p)

# Plot PMF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(k, pmf_values, color='blue', alpha=0.6, label="PMF")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Probability")
plt.title("Binomial Distribution - PMF")
plt.legend()

# Plot CDF
plt.subplot(1, 2, 2)
plt.plot(k, cdf_values, color='red', label="CDF", linestyle="dashed")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Cumulative Probability")
plt.title("Binomial Distribution - CDF")
plt.legend()

plt.tight_layout()
plt.show()

# For continuous distribution
x = np.linspace(-4, 4, 100)
mu = np.mean(x)
sigma = np.std(x)

pdf_values = norm.pdf(x, mu, sigma)
cdf_values = norm.cdf(x, mu, sigma)

# Plot PDF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, pdf_values, color='blue', label="PDF")
plt.xlabel("X Value")
plt.ylabel("Probability Density")
plt.title("Normal Distribution - PDF")
plt.legend()

# Plot CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf_values, color='red', linestyle="dashed", label="CDF")
plt.xlabel("X Value")
plt.ylabel("Cumulative Probability")
plt.title("Normal Distribution - CDF")
plt.legend()

plt.tight_layout()
plt.show()

Practical - 6: Write a program to implement t test, normal test, F test---------------------------------------------------------------------------------------------------------------------------------------
from scipy.stats import ttest_1samp, f_oneway
import numpy as np

# T-test
# Sample data (exam scores of a class)
print("....T-test.....")

print("""H0: The average exam score is significantly different from the population mean.
H1: There is no significant difference in the average exam score.""")
sample_scores = [75, 82, 88, 78, 95, 89, 92, 85, 88, 79]

# Population mean (hypothetical mean of all students' scores)
population_mean = 60

# T-statistic: This value indicates how many standard deviations the sample mean is away from the hypothesized population mean.
t_statistic, cal_value = ttest_1samp(sample_scores, population_mean)

# Output the results
print(f"t-statistic: {t_statistic}")
print(f"cal-value: {cal_value}")

# Check if the result is statistically significant (using a common significance level of 0.05)
if cal_value < 0.05:
    print("Accept H0: The average exam score is significantly different from the population mean.")
else:
    print("""Reject H0 
    Accept H1: There is no significant difference in the average exam score.
    """)
# F-test
print("....F-test.....")
data1 = [5, 8, 6, 7, 8]
data2 = [10, 20, 15, 18, 16]
f_stat, cal_value = f_oneway(data1, data2)

print("F-test: F-statistic =", f_stat, ", cal-value =", cal_value)

if cal_value < 0.05:
    print("Accept H0: Significant difference between variances.")
else:
    print("""Reject H0
    Accept H1: No significant difference between variances.""")

Practical - 7: Write a program to implement Analysis of Variance-----------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import f_oneway, f

print("""H0: There is no significant difference between the class scores of the department.
H1: There is a significant difference between the class scores of the department.""")
IT = np.array([23, 20, 22, 21, 24])
DS = np.array([30, 32, 29, 31, 33])
CS = np.array([40, 42, 41, 39, 43])

all_data = np.concatenate([IT, DS, CS])
mean_data = np.mean(all_data)

mean1 = np.mean(IT)
mean2 = np.mean(DS)
mean3 = np.mean(CS)

#between sample
ssc = len(IT)*(mean1 - mean_data)**2 + len(DS)*(mean2 - mean_data)**2 + len(CS)*(mean3 - mean_data)**2
# OR ssc = [len(g)*(np.mean(g) - mean_data)**2 for g in [IT, DS, CS]]
print(f"Between Sample (SSC):{ssc}")
degree_of_freedom_c = len([IT, DS, CS]) - 1 # c-1
MSC = ssc / degree_of_freedom_c
print(f"Mean Square (MSC):{MSC}")

#within sample
sse = sum((x - mean1)**2 for x in IT) + sum((x - mean2)**2 for x in DS) + sum((x - mean3)**2 for x in CS)
# OR sse = sum(sum((g - np.mean(g)) ** 2) for g in [IT, DS, CS])
print(f"Within Sample (SSE):{sse}")
degree_of_freedom_e = len(all_data) - len([IT, DS, CS]) # n-c
MSE = sse / degree_of_freedom_e
print(f"Mean Square (MSE):{MSE}")

# Total
sst = ssc + sse
df_sst = len(all_data) - 1
print("Sum of square total (sst):", sst)
print("degree of freedom:", df_sst)


varience = MSC / MSE
print(f"Varience:{round(varience, 3)}")
cal_value = f.sf(varience, degree_of_freedom_c, degree_of_freedom_e)
print(f"cal-value (Manual Calculation):{cal_value}")

alpha = 0.05
if cal_value < alpha:
    print("Accept H0: There is no significant difference between the class scores of the department.")
else:
    print("""Reject H0
    Accept H1: There is a significant difference between the class scores of the department.""")

Practical - 8: Write a program to implement Non parametric tests- I,II--------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import wilcoxon

# Sign test 
print("H0: There is no significant difference between median temperature which is below 30 degree")
print("H0: There is a significant difference between median temperature which is below 30 degree")

temperatures = [28, 29, 31, 30, 32, 33, 27]
median = int(np.median(temperatures))
print(f"Question:{data}, HO: meadian == {median}")
data_n = [i - median for i in temperatures]
print(f"After substracting by {median}, we get:{data_n}")

remove_zeros = [i for i in data_n if i != 0]
print(remove_zeros)
positive_T = len([i for i in remove_zeros if i > 0])
print(f"T+:{positive_T}")
negative_T = len([i for i in remove_zeros if i < 0])
print(f"T-:{negative_T}")

T = min([positive_T, negative_T])
print(f"T = {T}")

if T < 0.05:
    print("H0: There is no significant difference between median temperature which is below 30 degree")
else:
    print("H0: There is a significant difference between median temperature which is below 30 degree")

# Wilcoxon signed-rank test
# Sample data: Before and After measurements
before = [72, 65, 78, 71, 69, 68, 74, 70]
after  = [75, 67, 76, 73, 70, 69, 76, 71]

# Perform the Wilcoxon signed-rank test
statistic, p_value = wilcoxon(before, after)

# Output results
print("Wilcoxon Signed-Rank Test")
print(f"Test Statistic: {statistic}")
print(f"P-Value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis — there is a significant difference.")
else:
    print("Fail to reject the null hypothesis — no significant difference.")


Practical - 9: Write a program to implement Kruskal-Walis tests------------------------------------------------------------------------------------------------------------------------------------------
from scipy.stats import kruskal

group1 = [7, 14, 14, 13, 12, 9, 6, 14, 12, 8]
group2 = [15, 17, 13, 15, 15, 13, 9, 12, 10, 8]
group3 = [6, 8, 8, 9, 5, 14, 13, 8, 10, 9]

value, cal_value = kruskal(group1, group2, group3)
print(value)
print(cal_value)

if cal_value > 0.05:
    print("Accept H0")
else:
    print("Reject H0")

Practical - 10: Write a program to implement Wilcoxon‟s signed rank test----------------------------------------------------------------------------------------------------------------------------------
import numpy as np

def wilcoxon_signed_rank(before, after):
    # Step 1: Differences
    diff = np.array(after) - np.array(before)
    
    # Step 2: Remove zero differences
    non_zero = diff != 0
    diff = diff[non_zero]
    
    # Step 3: Absolute differences and ranks
    abs_diff = np.abs(diff)
    ranks = abs_diff.argsort().argsort() + 1  # simple rank (not handling ties for simplicity)
    
    # Step 4: Add back signs
    signed_ranks = ranks * np.sign(diff)
    
    # Step 5: Sum of positive and negative ranks
    W_pos = sum(signed_ranks[signed_ranks > 0])
    W_neg = -sum(signed_ranks[signed_ranks < 0])
    
    # Step 6: Test statistic
    W = min(W_pos, W_neg)

    print("Signed Ranks:", signed_ranks)
    print("W+ (positive ranks sum):", W_pos)
    print("W- (negative ranks sum):", W_neg)
    print("Wilcoxon Test Statistic (W):", W)
    
    # Note: To get p-value or compare to critical value, use a table or stats library

# Example data
before = [72, 68, 75, 80, 85, 90, 88, 76, 74, 79]
after  = [70, 66, 74, 77, 82, 88, 85, 75, 73, 76]

wilcoxon_signed_rank(before, after)

Practical - 11: Write a program to implement Time Series Analysis and Forecasting----------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

dates = pd.date_range(start='2020-01-01', periods=7, freq='M')
xx = [1, 2, 3, 4, 5, 6, 7]
trend = np.random.normal(0, 10, 7)

plt.scatter(dates, trend)
plt.plot(dates, trend, color = "green")
plt.show()

# Moving avg 
listt = [15, 21,30, 36, 42, 46, 50,56, 63, 70, 74, 82, 102]
df = pd.DataFrame(listt)

temp = df.rolling(window = 3).sum()
temp2 = df.rolling(window = 5).sum()

plt.plot(listt)
plt.plot(temp)
plt.plot(temp2)
plt.show()

# least square method 
years =  np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021])
production =  np.array([40, 42, 44, 46, 50, 53, 55])
assumed_mean = 2018

x = years - assumed_mean
y = production

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

b = sum_xy / sum_x2 # b= ∑xy / ∑x2 (Slope)
a = np.mean(y) - (b * np.mean(x))

print(f"Least Square Regressions Line: y = {a} + {b}x")

prediction_y = a + b * x

plt.scatter(years, production, color = "blue", label = "Actual Data")
plt.plot(years, prediction_y, color = "red", linestyle = "dashed", 
         label = "Regression Line")
plt.xlabel("Year")
plt.ylabel("Production")
plt.title("Least square Regression Line (production vs year)")
plt.legend()
plt.grid()
plt.show()
