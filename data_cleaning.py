import pandas as pd
pd.set_option('display.max_columns', 55)


df = pd.read_csv('all_listings.csv')
df = df.drop(columns='Unnamed: 0')


df.head(2)


df = df.drop(columns=['creation_tsz', 'ending_tsz', 'original_creation_tsz', 'last_modified_tsz', 'sku', 'state_tsz', 'shipping_template_id', 'item_weight', 'item_weight_unit', 'item_length', 'item_width', 'item_height', 'item_dimensions_unit', 'non_taxable', 'is_customizable', 'is_digital', 'file_data', 'should_auto_renew', 'has_variations', 'used_manufacturer', 'is_vintage', 'featured_rank', 'is_private', 'who_made', 'style'])


df.to_csv('listings.csv')


df.isnull().sum()
df = df.dropna(subset = ['user_id'])
df.to_csv('listings.csv')


len(df)
len(df.user_id.value_counts())
len(df.taxonomy_id.value_counts())
len(df.listing_id.value_counts())
len(df.category_id.value_counts())
len(df.category_path_ids.value_counts())

df[df.groupby('taxonomy_path')['taxonomy_path'].transform('size') >= 200]

df.listing_id.value_counts()

df = df.drop_duplicates(subset='listing_id')
df.to_csv('unique_listings.csv')
