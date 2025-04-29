# 币安数据获取工具

这个工具用于从币安获取比特币历史价格数据、交易量和订单簿信息。

## 功能

- 获取历史K线数据（OHLCV）
- 获取当前订单簿数据
- 保存数据到CSV文件
- 生成价格和交易量图表

## 安装依赖

在使用此脚本之前，请确保安装了所需的依赖：

```bash
pip install python-binance pandas matplotlib python-dateutil
```

## 使用方法

1. 打开 `binance_data_fetcher.py` 文件
2. 根据需要修改以下参数：
   - `symbol`: 交易对，默认为 'BTCUSDT'
   - `interval`: K线间隔，默认为 '15m'（15分钟）
   - `start_date`: 开始日期，默认为 '2023-06-01'
   - `end_date`: 结束日期，默认为 '2023-12-31'
3. 运行脚本：

```bash
python binance_data_fetcher.py
```

## 输出

脚本将生成以下输出：

1. K线数据CSV文件：`output/binance_data/BTCUSDT_15m_2023-06-01_to_2023-12-31.csv`
2. 当前订单簿数据CSV文件：
   - `output/binance_data/BTCUSDT_bids_[timestamp].csv`
   - `output/binance_data/BTCUSDT_asks_[timestamp].csv`
3. 价格和交易量图表：`BTCUSDT_price_volume_chart.png`

## 币安API可获取的数据类型

1. K线数据 (OHLCV) - 可获取历史数据
   - 开盘价、最高价、最低价、收盘价
   - 交易量 (基础资产和报价资产)
   - 交易笔数
   - 买方成交量 (基础资产和报价资产)
2. 订单簿数据 - 仅当前数据，历史数据有限制
   - 买单 (价格和数量)
   - 卖单 (价格和数量)
3. 最新成交 - 可获取近期成交
4. 24小时价格变动统计
5. 交易对信息 (价格精度、数量精度等)

## 注意事项

- 币安API对请求频率有限制，请避免频繁调用
- 历史订单簿数据通常不可直接获取，只能获取当前订单簿状态
- 对于大量历史数据，可能需要分批获取以避免超时

## 扩展功能

如需获取更多类型的数据或实现其他功能，可以参考币安API文档：
https://binance-docs.github.io/apidocs/spot/cn/
