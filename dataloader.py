from fredapi import Fred
import pandas as pd


api_key = ''
def load_data():
    """
    Load data from FRED API
    :return: a pandas dataframe with the loaded data
    """
    fred = Fred(api_key)
    fred_ids = [
        "BAMLH0A0HYM2",
        "BAMLH0A1HYBB",
        "BAMLH0A0HYM2EY",
        "BAMLH0A1HYBBEY",
        "DGS10",
        "DGS3",
        "DGS1",
        "DGS5",
        "VIXCLS",
        "DEXUSEU",
        "DCOILWTICO",
        "DEXJPUS",
        "NASDAQCOM",
    ]
    data = []
    for fred_id in fred_ids:
        series = fred.get_series(fred_id, observation_start="1990-01-01")
        data.append(series)
        print(data[-1])
    data = pd.DataFrame(data).T
    data.columns = [
        "Non Investment Grade Short Term",
        "Investment Grade Short Term",
        "Non Investment Grade Short Term Excess Return",
        "Investment Grade Short Term Excess Return",
        "10-Year Treasury",
        "3-Year Treasury",
        "1-Year Treasury",
        "5-Year Treasury",
        "VIX",
        "USD to EUR Exchange Rate",
        "Crude Oil Prices",
        "USD to JPY Exchange Rate",
        "NASDAQ Composite Index",
    ]


    return data.dropna()
def save_csv(data):
    """
    Save data to a csv file with the date as the index
    :param data: a pandas dataframe
    :return: None
    """
    data.to_csv("fred_data.csv", index=True)
    return None
if __name__ == '__main__':
    data = load_data()
    print(data.to_string())
    save_csv(data)
