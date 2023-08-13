from alchemy_api import AlchemyApi
import pandas as pd
import numpy as np
import pickle
import random
from web3 import Web3
import concurrent.futures
import re

alchemy = AlchemyApi()

filtered_may_june_addresses = pd.read_pickle('data/pickle_files/filtered_may_june_addresses.pkl')

def get_address_type(address, api_keys, alchemy_url):
    check_sum_address = Web3.to_checksum_address(address)
    api_key = api_key_rotation(api_keys)
    full_url = alchemy_url + api_key
    w3 = Web3(Web3.HTTPProvider(full_url))
    response = w3.eth.get_code(check_sum_address)
    return response.hex() == '0x'

def api_key_rotation(api_keys):
    index = random.randint(0, len(api_keys) - 1)
    return api_keys[index]

user_addresses = []
contract_addresses = []
api_keys = alchemy.get_api_keys()
alchemy_url = alchemy.get_api_url()

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    future_to_address = {executor.submit(get_address_type, address, api_keys, alchemy_url): address for address in filtered_may_june_addresses}
    for future in concurrent.futures.as_completed(future_to_address):
        address = future_to_address[future]
        try:
            is_user_address = future.result()
            if is_user_address:
                user_addresses.append(address)
            else:
                contract_addresses.append(address)
        except TypeError as e:
            continue

with open('data/pickle_files/may_june_user_addresses.pkl', 'wb') as f:
    pickle.dump(user_addresses, f)

with open('data/pickle_files/may_june_contract_addresses.pkl', 'wb') as f:
    pickle.dump(contract_addresses, f)