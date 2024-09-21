import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Converting to DataFrame for better visualization
# column_names = [f'feature_{i}' for i in range(X.shape[1])]
column_names = iris.feature_names
df = pd.DataFrame(X, columns=column_names)
df['target'] = y

print("Original Dataset:")
print(df.head())

# Applying Chi-Square feature selection and
# Selecting top k features
k = 4 
chi2_selector = SelectKBest(chi2, k=k)
X_new = chi2_selector.fit_transform(X, y)

selected_features = df.columns[:-1][chi2_selector.get_support()]
print("\nSelected Features:")
print(selected_features)
