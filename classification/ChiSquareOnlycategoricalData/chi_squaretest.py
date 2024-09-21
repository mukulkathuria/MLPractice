import pandas as pd
from scipy.stats import chi2_contingency


df = pd.read_csv('MaleFemaleCarTypeTarget.csv')
df = df.drop(columns='Unnamed: 0')

data = pd.crosstab(df['target'],[df['category'], df['cardatatype']])
# print(data.describe())
chi2, p, dof, expected = chi2_contingency(data)

print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:", expected)

if p < 0.05:
    print("You don't have to remove that features")
else:
    print("You can remove that features")