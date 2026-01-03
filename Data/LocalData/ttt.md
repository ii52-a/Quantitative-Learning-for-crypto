# 数据库
### ORDER BY返回数据排序
- 默认为ASC升序
- DESC为降序
### LIMIT 限制符合条件的返回条数
## 获取所有数据
- SELECT * FROM kline
- pd直接操作转化为dataframe: pd=read_sql_query('SELECT - FROM kline',连接)
## 获取部分数据
- SELECT ?,? FROME kline
- rows = cursor.fetchall()
## 查询最新一条信息
- SELECT * FROM kline ORDER BY timestamp DESC LIMIT 1
- row = cursor.fetchone()
## 按时间区间查询
- SELECT * FROM kline WHERE timestamp >= ? AND timestamp<= ? ORDER BY timestamp ASC
- rows = cursor.fetchall()

## 获取某项最大数据
- SELECT MAX(timestamp) FROM kline
- max_ts = cursor.fetchone()[0]
