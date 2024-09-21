from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=5, n_classes=4, n_clusters_per_class=4,
    random_state=42
)

f_stats , p_value = f_classif(X,y)
print("F values",f_stats)
print("P Value: ", p_value)

# After checking F values we find out highest is greater so 7 features is impacting the results
selectedFeatures = np.argsort(f_stats)[-7:]
features = X[:, selectedFeatures]

z_score = np.abs(stats.zscore(features))
df=features[(z_score < 3).all(axis=1)]
print(df)
# scatter = sns.scatterplot(x=features)
# fig = scatter.get_figure()
# fig.savefig('figures/classificationanovatest001.png', dpi=1200)


# f_stats , p_value = f_oneway(X)
# print(f_stats)
# print("P value", p_value)



# visualising classification
# from matplotlib.colors import ListedColormap
# x1, x2 = np.meshgrid(np.arange(start = features[:, 0].min() - 1, stop = features[:, 0].max() + 1, step  =0.01),  
# np.arange(start = features[:, 1].min() - 1, stop = features[:, 1].max() + 1, step = 0.01))  
# plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
# alpha = 0.75, cmap = ListedColormap(('red','green' )))  
# plt.xlim(x1.min(), x1.max())  
# plt.ylim(x2.min(), x2.max())  
# for i, j in enumerate(np.unique(y)):  
#     plt.scatter(features[y == j, 0], features[y == j, 1],  
#         c = ListedColormap(('red', 'green'))(i), label = j)  
# plt.title('K-NN Algorithm (Training set)')  
# plt.xlabel('Age')  
# plt.ylabel('Estimated Salary')  
# plt.legend()  
# plt.show()  