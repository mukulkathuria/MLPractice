import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
df = pd.read_csv('../data/highskewed.csv')
df = df.drop(columns='Unnamed: 0')


count = 300
target = []
for column in df.columns:
    for j in range(len(df[column])):
        if df.loc[j, column] > 2.5:
            count = count +5
            target.append(count)
        else:
            target.append(0)

df['target'] = target
# print(df.describe())

z_score = np.abs(stats.zscore(df))
df=df[(z_score < 3).all(axis=1)]
# print(df.describe())


X = df['A']
Y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train = np.log(X_train)
# y_train = np.cbrt(y_train)
# y_train = stats.boxcox(y_train)
# print(X_train, y_train)

model = RandomForestRegressor(n_estimators=1000, max_depth=4)
model.fit(X_train.values.reshape(-1,1), y_train)
predict = model.predict(X_test.values.reshape(-1,1))
values = [value for value in predict]

# scatter = sns.scatterplot(x=X_test.values, y=y_test.values)
# plt.xlabel('A')
# plt.ylabel('Actual')
# fig = scatter.get_figure()
# fig.savefig('figures/knntest1scatterActual.png', dpi=1200)
# print("Saved")
mse = mean_squared_error(y_test, values)
# rmse = mean_squared_error(y_test, values)
mae = mean_absolute_error(y_test, values)
r2 = r2_score(y_test, values)

mape = np.mean(np.abs((y_test - values) / y_test)) * 100

print("MSE:", mse)
# print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)
print("MAPE:", mape)