import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
y_train = np.cbrt(y_train)
# y_train = stats.boxcox(y_train)



# histograms
# hist = sns.histplot(data=X_train, bins=20, kde=True)
# fig = hist.get_figure()
# fig.savefig('figures/linear1hist4Atwithlogintraining.png', dpi=1200)
# print("Saved")

# hist = sns.histplot(data=y_train, bins=20, kde=True)
# fig = hist.get_figure()
# fig.savefig('figures/linear1hist4targettwithlogintraining.png', dpi=1200)
# print("Saved")


# scatterplot
scatter = sns.scatterplot(x=X_train, y=y_train)
fig = scatter.get_figure()
fig.savefig('figures/linear1scatter6withlogintraining.png', dpi=1200)
print("Saved")



# check the best fit line
# reg = sns.regplot(x="A", y="target", data=df);
# fig = reg.get_figure()
# fig.savefig('figures/linear2bestfit.png', dpi=1200)
# print("Saved")



# model = LinearRegression()
# model.fit(X_train.values.reshape(-1,1), y_train)

# predict = model.predict(X_test.values.reshape(-1,1))
# print(predict)