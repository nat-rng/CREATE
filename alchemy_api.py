import copy

class AlchemyApi:
    def __init__(self):
        self._api_key = ["1DEtj3Pe9LuLSHuODuWoT9M2v3KacR-L","aRbZxZwQA5D39RyckDXjA_z2fMh_tVQt", "zH7u-sd4JM3NWvtJBL1dXm6tMNtLqw6Z", "E_zhA7QVnQXpQADcNfqfilkOTzqsLh8W",
                         "hW2uvIrUWVU7SQkTPtabk3tU8o6c4Q2j", "8hXDABOuw1X0OJcHguG7SfvBpoOqe3Gr", "Wo40h2D23XmQtALGgiBWhp0a35GYJTO0", "XlnieXQwiTmui3TZq07YAN0CgHcIe2IN",
                         "AvS-2ltxE6Tr11hII9NOziRiu2srFE6S", "iEGBiZHSxm4IZZ-RDs6syria_wi4fwY6", "FZkiLhCC2gaTQrBSm1SQx6ssaAH7DfLH", "mr926zUYwHychvJ6jBejTZ4R2XtKzL0i",
                         "R_UC4pKshmfRnby2uiHIYTsnp5CAnNUP", "_TYqZcsHIpRF-un1fmYvLS-lDlL5TmdB", "P9rTu0vfTYEMVyENyf3OSP_8FWY35CbF", "MvtOPJi-iCtk1JbngjpjYIjRAmZpyK-Y"]
        self._api_url = 'https://eth-mainnet.alchemyapi.io/v2/'
        self._asset_transfers = {
                                "id": 1,
                                "jsonrpc": "2.0",
                                "method": "alchemy_getAssetTransfers",
                                "params": [
                                    {
                                    "category": [
                                        "erc20",
                                        "external",
                                        "internal",
                                        "erc721",
                                        "erc1155",
                                        "specialnft"
                                    ],
                                    "withMetadata": True,
                                    "excludeZeroValue": False,
                                    }
                                ]
                            }
        self._transfer_receipt = {
                                    "id": 1,
                                    "jsonrpc": "2.0",
                                    "method": "alchemy_getTransactionReceipts",
                                    "params": []
                                }

    def get_api_keys(self):
        return self._api_key

    def get_api_url(self):
        return self._api_url

    def create_asset_transfers(self):
        asset_transfers_body = copy.deepcopy(self._asset_transfers)
        return asset_transfers_body

    def set_tx_address(self, asset_transfers_body, address_from=None, address_to=None):
        if address_from:
            asset_transfers_body["params"][0]['fromAddress'] = address_from
        if address_to:
            asset_transfers_body["params"][0]['toAddress'] = address_to
        return asset_transfers_body

    def set_tx_block_range(self, asset_transfers_body, block_from, block_to):
        asset_transfers_body["params"][0]['fromBlock'] = block_from
        asset_transfers_body["params"][0]['toBlock'] = block_to
        return asset_transfers_body

    def set_tx_pagination(self, asset_transfers_body, page_key):
        asset_transfers_body["params"][0]['pageKey'] = page_key
        return asset_transfers_body
    
    def create_transfer_receipt(self):
        transfer_receipt_body = copy.deepcopy(self._transfer_receipt)
        return transfer_receipt_body
    
    def set_receipt_block(self, transfer_receipt_body, block_number):
        blocknum_dict = {"blockNumber": block_number}
        transfer_receipt_body["params"].append(blocknum_dict)
        return transfer_receipt_body

    def convert_block_to_hex(self, block_number):
        return hex(block_number)

