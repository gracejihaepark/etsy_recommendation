import pandas as pd
pd.set_option('display.max_columns', 999)


df = pd.read_csv('unique_listings.csv')
df = df.drop(columns='Unnamed: 0')

df


df.describe()

df.category_id.value_counts()

df.taxonomy_id.value_counts()

df.isnull().sum()


dfdummies = pd.get_dummies(df)

dfdummies



import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
ax = sns.countplot(x='taxonomy_path', data=df[df.groupby('taxonomy_path')['taxonomy_path'].transform('size') >= 100])



df[df['views']==df['views'].max()]
