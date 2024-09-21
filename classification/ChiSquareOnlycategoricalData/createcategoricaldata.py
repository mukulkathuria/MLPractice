import random
import pandas as pd
# current_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(current_dir)

# from renamedata.gender import data

randomtarget = [0,1]
data = ["Male", "Female"]
cardatatype = ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible"]

dataset = {}

for i in range(10000):
    if 'category' in dataset:
        dataset['category'].append(random.choice(data))
    else:
        dataset['category'] = [random.choice(data)]
    if 'cardatatype' in dataset:
        dataset['cardatatype'].append(random.choice(cardatatype))
    else:
        dataset['cardatatype'] = [random.choice(cardatatype)]
    if 'target' in dataset:
        dataset['target'].append(random.choice(randomtarget))
    else:
        dataset['target'] = [random.choice(randomtarget)]

df = pd.DataFrame(dataset)
df.to_csv('MaleFemaleCarTypeTarget.csv')