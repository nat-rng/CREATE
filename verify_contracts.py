from alchemy_api import AlchemyApi
import pandas as pd
import pickle
import random
from web3 import Web3
import concurrent.futures

# Helper function to split a list into n parts
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

alchemy = AlchemyApi()

test_df = pd.read_parquet('data/parquet_files/test_df.parquet')
train_two_month_df = pd.read_parquet('data/parquet_files/train_two_month_df.parquet')

check_addresses = test_df['address'].unique().tolist() + train_two_month_df['address'].unique().tolist()
check_addresses = list(set(check_addresses))

user_addresses = pd.read_pickle('data/pickle_files/user_addresses.pkl')
contract_addresses = pd.read_pickle('data/pickle_files/contract_addresses.pkl')

check_addresses = list(set(check_addresses) - set(user_addresses) - set(contract_addresses))

chunk_size = len(check_addresses) // 5

address_chunks = list(chunk_list(check_addresses, chunk_size))

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

api_keys = alchemy.get_api_keys()
alchemy_url = alchemy.get_api_url()

for i, chunk in enumerate(address_chunks):
    print(f"Processing chunk {i + 1} of {len(address_chunks)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_address = {executor.submit(get_address_type, address, api_keys, alchemy_url): address for address in chunk}
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

    # Save the updated lists after processing each chunk
    with open('data/pickle_files/user_addresses.pkl', 'wb') as f:
        pickle.dump(user_addresses, f)

    with open('data/pickle_files/contract_addresses.pkl', 'wb') as f:
        pickle.dump(contract_addresses, f)

    print(f"Chunk {i + 1} processed and pickle files updated.")
