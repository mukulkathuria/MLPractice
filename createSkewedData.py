import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import lognorm, pareto, gamma, uniform

# Highly skewed data (log-normal)
skewed_data = lognorm.rvs(s=2, size=1000)
data = {
    'A': skewed_data
}
df = pd.DataFrame(data=data)
df.to_csv("data/highskewed.csv")
# histo = sns.histplot(data=df, kde=True, color='orange', bins=50)
# fig = histo.get_figure()
# fig.savefig('figures/testhisto.png', dpi=1200)
