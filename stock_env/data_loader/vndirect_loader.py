from datetime import datetime
import requests
import pandas as pd
from typing import List, Dict, Tuple, Union

API_VNDIRECT = 'https://finfo-api.vndirect.com.vn/v4/stock_prices/'
HEADERS = {'content-type': 'application/x-www-form-urlencoded', 'User-Agent': 'Mozilla'}


class VNDDataLoader:
    def __init__(
        self, 
        symbols: List[str], 
        start: str, 
        end: str
    ):
        self.symbols = symbols
        self.start = start
        self.end = end

    def download(self):
        stock_datas = []
        if not isinstance(self.symbols, list):
            symbols = [self.symbols]
        else:
            symbols = self.symbols

        for symbol in symbols:
            stock_datas.append(self._download(symbol))

        data = pd.concat(stock_datas, axis=0)
        return data
    
    def _download(self, symbol):
        start_date = self.start
        end_date = self.end
        query = 'code:' + symbol + '~date:gte:' + start_date + '~date:lte:' + end_date
        delta = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        params = {
            "sort": "date",
            "size": delta.days + 1,
            "page": 1,
            "q": query
        }
        res = requests.get(API_VNDIRECT, params=params, headers=HEADERS)
        data = res.json()['data']
        data = pd.DataFrame(data)
        data = data.sort_values(by='date')
        data = data.reset_index(drop=True)
        data['volume'] = data['nmVolume'] + data['ptVolume']
        data = data[['date', 'code', 'open', 'high', 'low', 'close', 
                    'adOpen', 'adHigh', 'adLow', 'adClose', 'volume']]
        data['date'] = pd.to_datetime(data['date'])
        return data