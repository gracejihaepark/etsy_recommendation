import pandas as pd
from etsy_py.api import EtsyAPI
import config
api = EtsyAPI(api_key = config.apikey)

uri = 'listings/active'
url_params = {
                'limit': 100
            }


def api_call(uri, url_params):
    api = EtsyAPI(api_key = config.apikey)
    response = api.get(uri, params=url_params)
    listings_data = response.json()
    return listings_data


# all_listings = []
cur = 45000
while cur < 50100:
    url_params['offset'] = cur
    listings_data = api_call(uri, url_params)
    all_listings.extend(listings_data['results'])
    cur += 100


len(all_listings)


listings_df = pd.DataFrame.from_dict(all_listings)
listings_df.to_csv('all_listings.csv')
