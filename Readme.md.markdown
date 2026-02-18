---
jupyter:
  kaggle:
    accelerator: none
    dataSources:
    - datasetId: 9512716
      sourceId: 14870011
      sourceType: datasetVersion
    dockerImageVersionId: 31259
    isGpuEnabled: false
    isInternetEnabled: true
    language: python
    sourceType: notebook
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.12
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2026-02-17T16:14:02.220191Z\",\"iopub.status.busy\":\"2026-02-17T16:14:02.219819Z\",\"iopub.status.idle\":\"2026-02-17T16:14:08.741934Z\",\"shell.execute_reply\":\"2026-02-17T16:14:08.740853Z\",\"shell.execute_reply.started\":\"2026-02-17T16:14:02.220159Z\"}" trusted="true"}
``` python
!pip install mplfinance
```

::: {.output .stream .stdout}
    Collecting mplfinance
      Downloading mplfinance-0.12.10b0-py3-none-any.whl.metadata (19 kB)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (from mplfinance) (3.10.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (from mplfinance) (2.2.2)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (4.60.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (1.4.9)
    Requirement already satisfied: numpy>=1.23 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (2.0.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (26.0rc2)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (3.2.5)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mplfinance) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas->mplfinance) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas->mplfinance) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib->mplfinance) (1.17.0)
    Downloading mplfinance-0.12.10b0-py3-none-any.whl (75 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.0/75.0 kB 2.8 MB/s eta 0:00:00
    plfinance
    Successfully installed mplfinance-0.12.10b0
:::
:::

::: {.cell .code execution_count="3" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution="{\"iopub.execute_input\":\"2026-02-17T16:14:14.961991Z\",\"iopub.status.busy\":\"2026-02-17T16:14:14.961624Z\",\"iopub.status.idle\":\"2026-02-17T16:14:14.992682Z\",\"shell.execute_reply\":\"2026-02-17T16:14:14.991724Z\",\"shell.execute_reply.started\":\"2026-02-17T16:14:14.961953Z\"}" trusted="true"}
``` python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import mplfinance as mpf

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14,6)
```
:::

::: {.cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2026-02-17T16:17:41.845669Z\",\"iopub.status.busy\":\"2026-02-17T16:17:41.845318Z\",\"iopub.status.idle\":\"2026-02-17T16:17:43.377318Z\",\"shell.execute_reply\":\"2026-02-17T16:17:43.376322Z\",\"shell.execute_reply.started\":\"2026-02-17T16:17:41.845636Z\"}" trusted="true"}
``` python
# =====================================
# 2. Load Dataset
# =====================================
df_15m = pd.read_csv("/kaggle/input/forex-eurusd-dataset/ohlc_15m.csv", index_col=0, parse_dates=True )
df_1h = pd.read_csv("/kaggle/input/forex-eurusd-dataset/ohlc_1h.csv", index_col=0, parse_dates=True )
df_4h = pd.read_csv("/kaggle/input/forex-eurusd-dataset/ohlc_4h.csv", index_col=0, parse_dates=True )
df_1d = pd.read_csv("/kaggle/input/forex-eurusd-dataset/daily.csv", index_col=0, parse_dates=True )
```
:::

::: {.cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2026-02-17T16:17:56.989774Z\",\"iopub.status.busy\":\"2026-02-17T16:17:56.989436Z\",\"iopub.status.idle\":\"2026-02-17T16:17:57.022246Z\",\"shell.execute_reply\":\"2026-02-17T16:17:57.021203Z\",\"shell.execute_reply.started\":\"2026-02-17T16:17:56.989743Z\"}" trusted="true"}
``` python
# =====================================
# 3. Data Overview
# =====================================
def overview(df, name):
    print(f"--- {name} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print(df.head(), "\n")

overview(df_1d, "1D Data")
overview(df_4h, "4H Data")
overview(df_1h, "1H Data")
overview(df_15m, "15M Data")
```

::: {.output .stream .stdout}
    --- 1D Data ---
    Shape: (1497, 4)
    Columns: ['open', 'high', 'low', 'close']
    Missing values:
     open     0
    high     0
    low      0
    close    0
    dtype: int64
                                  open     high      low    close
    Timestamp                                                    
    2020-01-29 22:00:00+00:00  1.10090  1.10392  1.10070  1.10313
    2020-01-30 22:00:00+00:00  1.10311  1.10953  1.10170  1.10942
    2020-02-02 22:00:00+00:00  1.10918  1.10919  1.10362  1.10599
    2020-02-03 22:00:00+00:00  1.10596  1.10641  1.10330  1.10444
    2020-02-04 22:00:00+00:00  1.10442  1.10478  1.09937  1.09989 

    --- 4H Data ---
    Shape: (9051, 5)
    Columns: ['open', 'high', 'low', 'close', 'is_day_start']
    Missing values:
     open            0
    high            0
    low             0
    close           0
    is_day_start    0
    dtype: int64
                                  open     high      low    close  is_day_start
    Timestamp                                                                  
    2020-01-29 22:00:00+00:00  1.10090  1.10177  1.10072  1.10111          True
    2020-01-30 02:00:00+00:00  1.10112  1.10173  1.10112  1.10137         False
    2020-01-30 06:00:00+00:00  1.10136  1.10213  1.10070  1.10191         False
    2020-01-30 10:00:00+00:00  1.10190  1.10317  1.10114  1.10255         False
    2020-01-30 14:00:00+00:00  1.10254  1.10380  1.10155  1.10376         False 

    --- 1H Data ---
    Shape: (36173, 5)
    Columns: ['open', 'high', 'low', 'close', 'market_open']
    Missing values:
     open           0
    high           0
    low            0
    close          0
    market_open    0
    dtype: int64
                                  open     high      low    close  market_open
    Timestamp                                                                 
    2020-01-29 22:00:00+00:00  1.10090  1.10118  1.10072  1.10097         True
    2020-01-29 23:00:00+00:00  1.10101  1.10177  1.10089  1.10139        False
    2020-01-30 00:00:00+00:00  1.10138  1.10171  1.10130  1.10130        False
    2020-01-30 01:00:00+00:00  1.10131  1.10156  1.10111  1.10111        False
    2020-01-30 02:00:00+00:00  1.10112  1.10133  1.10112  1.10122        False 

    --- 15M Data ---
    Shape: (146902, 5)
    Columns: ['open', 'high', 'low', 'close', 'market_open']
    Missing values:
     open           0
    high           0
    low            0
    close          0
    market_open    0
    dtype: int64
                                  open     high      low    close  market_open
    Timestamp                                                                 
    2020-01-29 00:00:00+00:00  1.10223  1.10226  1.10204  1.10220        False
    2020-01-29 00:15:00+00:00  1.10221  1.10245  1.10216  1.10245        False
    2020-01-29 00:30:00+00:00  1.10246  1.10277  1.10245  1.10254        False
    2020-01-29 00:45:00+00:00  1.10255  1.10258  1.10223  1.10227        False
    2020-01-29 01:00:00+00:00  1.10228  1.10228  1.10210  1.10210        False 
:::
:::

::: {.cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2026-02-17T16:18:36.140740Z\",\"iopub.status.busy\":\"2026-02-17T16:18:36.140400Z\",\"iopub.status.idle\":\"2026-02-17T16:18:36.217011Z\",\"shell.execute_reply\":\"2026-02-17T16:18:36.215790Z\",\"shell.execute_reply.started\":\"2026-02-17T16:18:36.140711Z\"}" trusted="true"}
``` python
# =====================================
# 4. Summary Statistics
# =====================================
def summary_stats(df, name):
    print(f"--- Summary Statistics for {name} ---")
    print(df.describe(), "\n")

summary_stats(df_1d, "1D Data")
summary_stats(df_4h, "4H Data")
summary_stats(df_1h, "1H Data")
summary_stats(df_15m, "15M Data")
```

::: {.output .stream .stdout}
    --- Summary Statistics for 1D Data ---
                  open         high          low        close
    count  1497.000000  1497.000000  1497.000000  1497.000000
    mean      1.111726     1.115851     1.107921     1.111787
    std       0.057589     0.057120     0.058024     0.057594
    min       0.959280     0.967090     0.953600     0.959150
    25%       1.074030     1.078400     1.070920     1.073990
    50%       1.098020     1.101380     1.093590     1.097990
    75%       1.164320     1.167940     1.161090     1.164480
    max       1.232520     1.234950     1.226570     1.232480 

    --- Summary Statistics for 4H Data ---
                  open         high          low        close
    count  9051.000000  9051.000000  9051.000000  9051.000000
    mean      1.111747     1.113263     1.110273     1.111757
    std       0.057568     0.057395     0.057708     0.057567
    min       0.955070     0.957640     0.953600     0.955080
    25%       1.074175     1.075485     1.072790     1.074165
    50%       1.097590     1.099170     1.096250     1.097600
    75%       1.164295     1.165575     1.163110     1.164360
    max       1.233830     1.234950     1.232220     1.233840 

    --- Summary Statistics for 1H Data ---
                   open          high           low         close
    count  36173.000000  36173.000000  36173.000000  36173.000000
    mean       1.111749      1.112465      1.111037      1.111750
    std        0.057553      0.057477      0.057620      0.057552
    min        0.953940      0.955930      0.953600      0.953920
    25%        1.074110      1.074700      1.073520      1.074130
    50%        1.097740      1.098440      1.097070      1.097750
    75%        1.164330      1.164870      1.163730      1.164340
    max        1.234010      1.234950      1.233360      1.234030 

    --- Summary Statistics for 15M Data ---
                    open           high            low          close
    count  146902.000000  146902.000000  146902.000000  146902.000000
    mean        1.111793       1.112136       1.111445       1.111794
    std         0.057563       0.057529       0.057595       0.057563
    min         0.953940       0.955190       0.953600       0.953920
    25%         1.073900       1.074190       1.073600       1.073890
    50%         1.097960       1.098320       1.097630       1.097970
    75%         1.164258       1.164520       1.163940       1.164258
    max         1.234680       1.234950       1.234330       1.234670 
:::
:::

::: {.cell .code execution_count="10" execution="{\"iopub.execute_input\":\"2026-02-17T16:21:09.914406Z\",\"iopub.status.busy\":\"2026-02-17T16:21:09.914004Z\",\"iopub.status.idle\":\"2026-02-17T16:21:10.950465Z\",\"shell.execute_reply\":\"2026-02-17T16:21:10.949508Z\",\"shell.execute_reply.started\":\"2026-02-17T16:21:09.914375Z\"}" trusted="true"}
``` python
# =====================================
# 5. Returns and Percentage Changes
# =====================================
for df in [df_1d, df_4h, df_1h, df_15m]:
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Plot Return Distribution
plt.figure(figsize=(12,5))
sns.histplot(df_15m['return'].dropna(), bins=100, kde=True)
plt.title("15M Returns Distribution")
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/3bf01535ad661620360d6feb2014f30f2eb2f2ff.png)
:::
:::

::: {.cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2026-02-17T16:22:24.547328Z\",\"iopub.status.busy\":\"2026-02-17T16:22:24.546961Z\",\"iopub.status.idle\":\"2026-02-17T16:22:25.641958Z\",\"shell.execute_reply\":\"2026-02-17T16:22:25.640834Z\",\"shell.execute_reply.started\":\"2026-02-17T16:22:24.547298Z\"}" trusted="true"}
``` python
# =====================================
# 6. Time Series Visualization
# =====================================
# Plot Close Price
plt.figure(figsize=(14,6))
plt.plot(df_1h['close'], label='1H Close')
plt.plot(df_4h['close'], label='4H Close')
plt.title("EUR/USD Close Price")
plt.legend()
plt.show()

# Candlestick chart for 1D data
mpf.plot(df_1d.tail(60), type='candle', style='yahoo', title='1D Candlestick')
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/15b13b7ade13339189149c21125864cef6d6b475.png)
:::

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/b828f37edf2d95edd0112c9794e2bd8168243c17.png)
:::
:::

::: {.cell .code execution_count="13" execution="{\"iopub.execute_input\":\"2026-02-17T16:23:04.012105Z\",\"iopub.status.busy\":\"2026-02-17T16:23:04.011692Z\",\"iopub.status.idle\":\"2026-02-17T16:23:05.595772Z\",\"shell.execute_reply\":\"2026-02-17T16:23:05.594786Z\",\"shell.execute_reply.started\":\"2026-02-17T16:23:04.012070Z\"}" trusted="true"}
``` python
# =====================================
# 7. Moving Averages
# =====================================
for df in [df_1h, df_4h]:
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['SMA_200'] = df['close'].rolling(200).mean()

# Plot SMA
plt.figure(figsize=(14,6))
plt.plot(df_1h['close'], label='Close')
plt.plot(df_1h['SMA_50'], label='SMA 50')
plt.plot(df_1h['SMA_200'], label='SMA 200')
plt.title("1H Close Price with Moving Averages")
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/9daa859aa028cddcb5771b4c84a407ef23b11c46.png)
:::
:::

::: {.cell .code execution_count="16" execution="{\"iopub.execute_input\":\"2026-02-17T16:24:29.789969Z\",\"iopub.status.busy\":\"2026-02-17T16:24:29.789625Z\",\"iopub.status.idle\":\"2026-02-17T16:24:30.568635Z\",\"shell.execute_reply\":\"2026-02-17T16:24:30.567464Z\",\"shell.execute_reply.started\":\"2026-02-17T16:24:29.789939Z\"}" trusted="true"}
``` python
# =====================================
# 8. Volatility Analysis
# =====================================
for df in [df_15m, df_1h]:
    df['volatility'] = df['return'].rolling(20).std()

plt.figure(figsize=(14,6))
plt.plot(df_1h['volatility'], label='1H Volatility (Rolling 20)')
plt.title("1H Rolling Volatility")
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/08fe11cdd5826d7e5074749e7756964e880865e1.png)
:::
:::

::: {.cell .code execution_count="17" execution="{\"iopub.execute_input\":\"2026-02-17T16:27:10.108566Z\",\"iopub.status.busy\":\"2026-02-17T16:27:10.108184Z\",\"iopub.status.idle\":\"2026-02-17T16:27:10.573757Z\",\"shell.execute_reply\":\"2026-02-17T16:27:10.572642Z\",\"shell.execute_reply.started\":\"2026-02-17T16:27:10.108479Z\"}" trusted="true"}
``` python
# =====================================
# 9. Correlation Analysis
# =====================================
for df, name in zip([df_1h, df_4h], ['1H', '4H']):
    corr = df[['open','high','low','close','return']].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f"{name} Correlation Heatmap")
    plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/ab4aeadbc61b9c4c707f53cd555ae021f3bb27f6.png)
:::

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/85a58efb18e87ca2e4556379378a311a5a3e4964.png)
:::
:::

::: {.cell .code execution_count="18" execution="{\"iopub.execute_input\":\"2026-02-17T16:28:10.768844Z\",\"iopub.status.busy\":\"2026-02-17T16:28:10.768403Z\",\"iopub.status.idle\":\"2026-02-17T16:28:11.376592Z\",\"shell.execute_reply\":\"2026-02-17T16:28:11.375320Z\",\"shell.execute_reply.started\":\"2026-02-17T16:28:10.768800Z\"}" trusted="true"}
``` python
# =====================================
# 10. Candlestick Pattern Exploration (Basic Example)
# =====================================
# Detect simple bullish/bearish candlestick
def candle_pattern(row):
    if row['close'] > row['open']:
        return 'bullish'
    elif row['close'] < row['open']:
        return 'bearish'
    else:
        return 'doji'

df_1h['candle_type'] = df_1h.apply(candle_pattern, axis=1)
plt.figure(figsize=(6,4))
df_1h['candle_type'].value_counts().plot(kind='bar')
plt.title("1H Candlestick Types Count")
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/61883543eed40ebd8b6b4c84fc09e14736982e63.png)
:::
:::

::: {.cell .code execution_count="21" execution="{\"iopub.execute_input\":\"2026-02-17T16:30:06.516325Z\",\"iopub.status.busy\":\"2026-02-17T16:30:06.515954Z\",\"iopub.status.idle\":\"2026-02-17T16:30:06.767536Z\",\"shell.execute_reply\":\"2026-02-17T16:30:06.766118Z\",\"shell.execute_reply.started\":\"2026-02-17T16:30:06.516294Z\"}" trusted="true"}
``` python
# =====================================
# 11. Trend & Support/Resistance (Basic)
# =====================================
# Rolling max/min as dynamic support/resistance
df_1h['rolling_high'] = df_1h['high'].rolling(50).max()
df_1h['rolling_low'] = df_1h['low'].rolling(50).min()

plt.figure(figsize=(14,6))
plt.plot(df_1h['close'].tail(100), label='Close')
plt.plot(df_1h['rolling_high'].tail(100), label='50-Period High', linestyle='--')
plt.plot(df_1h['rolling_low'].tail(100), label='50-Period Low', linestyle='--')
plt.title("1H Close with Dynamic Support & Resistance")
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/349ac1004cfa4b28ca0c3d27d4159de66469bd9a.png)
:::
:::

::: {.cell .code execution_count="22" execution="{\"iopub.execute_input\":\"2026-02-17T16:30:46.262214Z\",\"iopub.status.busy\":\"2026-02-17T16:30:46.261738Z\",\"iopub.status.idle\":\"2026-02-17T16:30:46.697449Z\",\"shell.execute_reply\":\"2026-02-17T16:30:46.696565Z\",\"shell.execute_reply.started\":\"2026-02-17T16:30:46.262175Z\"}" trusted="true"}
``` python
# =====================================
# 12. Seasonality Analysis
# =====================================
df_1h['hour'] = df_1h.index.hour
hourly_returns = df_1h.groupby('hour')['return'].mean()
plt.figure(figsize=(10,4))
hourly_returns.plot(kind='bar')
plt.title("Average Return by Hour (1H Data)")
plt.show()

df_1h['weekday'] = df_1h.index.weekday
weekday_returns = df_1h.groupby('weekday')['return'].mean()
plt.figure(figsize=(8,4))
weekday_returns.plot(kind='bar')
plt.title("Average Return by Weekday (1H Data)")
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/4f9f7904f49508bede4e4acdb424d9136e9cb9cc.png)
:::

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/c5b636a1ca9d805026e5af7eeb64847d054c8344.png)
:::
:::

::: {.cell .code execution_count="24" execution="{\"iopub.execute_input\":\"2026-02-17T16:32:46.507547Z\",\"iopub.status.busy\":\"2026-02-17T16:32:46.507215Z\",\"iopub.status.idle\":\"2026-02-17T16:32:46.636876Z\",\"shell.execute_reply\":\"2026-02-17T16:32:46.635336Z\",\"shell.execute_reply.started\":\"2026-02-17T16:32:46.507515Z\"}" trusted="true"}
``` python
# =====================================
# 13. Outlier Detection
# =====================================
plt.figure(figsize=(8,4))
sns.boxplot(x=df_1d['return'])
plt.title("1D Returns Boxplot (Outlier Detection)")
plt.show()
```

::: {.output .display_data}
![](vertopal_99b474dd67df474c8191784a6b2512e0/f0b4cf674798a8afabb44a4380fb19407dc30944.png)
:::
:::
