import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
import paho.mqtt.publish as publish
from tsai.inference import load_learner
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

lookback_range = "90m"

query='''
from(bucket:"openhab_db/autogen")
        |> range(start: - ''' + lookback_range + ''', stop: now())
        |> filter(fn: (r) => r._measurement == "PowerConsumption")
        |> filter(fn: (r) => r._field == "value")
        |> rename(columns: {_value: "value", _time: "timestamp", _measurement: "item"})
        |> pivot(rowKey:["timestamp"], columnKey: ["_field"], valueColumn: "value")
'''

with InfluxDBClient(url="http://openhab:8086", token="openhab", debug=False) as client:
  df = client.query_api().query_data_frame(query=query)

df = df.loc[ df["item"] == "PowerConsumption", ["timestamp", "value" ] ]
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

df = df.resample("60s").max()
difference = df.diff(axis=0).dropna();
possible_ends = difference[ difference < -800 ].dropna()

if len(possible_ends) == 0:
    exit()

possible_matches = []
for possible_end in list(possible_ends.index):
    possible_start = possible_end - pd.Timedelta(minutes=90)
    series = df.loc[possible_start : possible_end]

    difference = series.diff(axis=0).dropna();
    idx = difference[ difference > 800 ].first_valid_index()
    series = series[idx:]

    # simple heuristic to prefilter:
    #   all samples must be present
    #   at least 1 kWh must be consumed in the timeframe
    #   the first 10 minutes must be > 1500 W

    kWh = series["value"].sum() / 60 / 1000

    if ( len(series) >= 60 and kWh > 1.5 and (series.iloc[0:10]["value"] > 1500).all() ):
        # restrict sample size, index by number (minute)
        series = series.reset_index(drop=True)

        # limit sample count and append additional samples if there aren't enough (repeat last value)
        last_value = df.iloc[-1]["value"]
        series = series[:90]
        series = series.reindex(range(90), fill_value=last_value)

        row = series.transpose()

        possible_matches.append( row )

if len(possible_matches) == 0:
    exit()

X = np.array(possible_matches)
clf = load_learner("tsai.pkl")
prob, _, pred = clf.get_X_preds(X)

for i in range(0, len(pred)):
    probability = prob[i][1]
    print(f"dishwasher probability: {probability:.3f}")

if (pred == "1").any(): 
    #publish.single("dishwasher", "done", hostname="192.168.10.222")
    print(f"found a dishwasher")
