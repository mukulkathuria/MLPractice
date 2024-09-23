import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans


df = pd.read_csv("../data/housing.csv")
# print(df.head())
# print(df.info())
# scatter = sns.scatterplot(data = df, x = 'longitude', y = 'latitude', hue = 'median_house_value')
# fig = scatter.get_figure()
# fig.savefig('figures/housinglonglatdataset001.png', dpi=1200)


X_train, X_test, y_train, y_test = train_test_split(df[['latitude', 'longitude']], df[['median_house_value']], test_size=0.2, random_state=42)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)


# kmeans = KMeans(n_clusters = 3, random_state = 42, n_init='auto')
# kmeans.fit(X_train_norm)

# scatter = sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
# fig = scatter.get_figure()
# fig.savefig('figures/housinglonglatdataset002.png', dpi=1200)

K = range(2, 8)
fits = []
score = []


for k in K:
    model = KMeans(n_clusters = k, random_state = 42, n_init='auto').fit(X_train_norm)
    fits.append(model)
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

checklineplot = sns.lineplot(x = K, y = score)
fig = checklineplot.get_figure()
fig.savefig('figures/housinglonglatdatasetKmeanscheckingbestKclusters.png', dpi=1200)

scatter = sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[4].labels_)
fig = scatter.get_figure()
fig.savefig('figures/housinglonglatdatasetBestKmeansClustersafterlookingKbest.png', dpi=1200)