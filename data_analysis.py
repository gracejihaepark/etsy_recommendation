import pandas as pd
pd.set_option('display.max_columns', 55)


df = pd.read_csv('listings.csv')
dftax = pd.read_csv('dftax.csv')
df = df.drop(columns='Unnamed: 0')
dftax = dftax.drop(columns='Unnamed: 0')

df
dftax




df[df['category_id'] == 69150433.0]

df.taxonomy_id.value_counts()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
ax = sns.countplot(x='taxonomy_path', data=dftax)






df.describe()
