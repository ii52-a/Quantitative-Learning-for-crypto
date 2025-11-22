# pd使用
## 切换数据类型
- data[cols]=data[cols].astype(float)
- pd.DataFrame 切换为DataFrame数据形式，通常key为首行
- set_index指定表头
- loc寻找
- data["寻找内容""]

# retry重试修饰器  tenacity库
- 可以设置函数自动重新获取
- retry修饰器, stop_after_attempt最大尝试, wait_fixed重试等待, retry_if_exception_type规定重试的异常原因
- 

#data.strf

# macd指标,macd(12,26,9)
- 指标为macd(12,26,9)的macd线
- macd:快速EMA<K线>(12)-慢速EMA<K线>(26)    DIF
- macd_signal:信号线,指9日平均,即EMA<DIF>(9)   DEA
- DIF>DEA 快线突破，上涨趋势可能
- DIF<DEA 快线死叉，下跌趋势可能
- macd_hist:macd-macd_signal,柱状图hist增大，趋势加强，反而反之