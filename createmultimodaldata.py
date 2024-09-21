import numpy as np
import pandas as pd 
import seaborn as sns


mu1, sigma1 = 0, 1
mu2, sigma2 = 5, 1
gaussian1 = np.random.normal(mu1, sigma1, 1000)
gaussian2 = np.random.normal(mu2, sigma2, 1000)
multimodal_data = np.concatenate([gaussian1, gaussian2])

data = {
    'A': multimodal_data
}
df = pd.DataFrame(data=data)
histo = sns.histplot(data=df, kde=True, color='orange')
fig = histo.get_figure()
fig.savefig('figures/testmultimodal.png', dpi=1200)
