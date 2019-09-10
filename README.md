# Usage-Demand-of-Bike-Share-in-Philadelphia
# 费城共享单车使用需求量分析
### 该项目旨在探索费城有桩共享单车项目 Indego 的出行情况所包含的时间、空间特征，及其与人口统计数据、天气情况的关联。
## 数据
### 项目中所使用的所有数据包括：
- 5-Year (2012-2017) American Community Survey (ACS) in Philadelphia (API through https://api.census.gov/) 美国人口普查数据（2012-2017）
- Liquor Violations in Philadelphia (API thourgh https://phl.carto.com) 费城违反酒精法规事件统计
- Bike-Share Trip Counts of Indego (https://www.rideindego.com/about/data/) Indego 出行数据
  - indego-trips-2018-q1.csv
  - indego-trips-2018-q2.csv
  - indego-trips-2018-q3.csv
  - indego-trips-2018-q4.csv

- Indego Stations (http://www.rideindego.com/stations/json/) Indego 站点信息
  - stations.json

- Weather Data (Average Daily Temperature & Precipitation) in Philadelphia in 2018 (https://www.wunderground.com/weather/us/pa/philadelphia) 费城2018年天气信息（日均天气、降水） 
  - weather.csv
## 研究结果
该项目主要分为两部分：探索分析（exploratory analysis)和回归分析（regression analysis)。在第一部分中，通过对出行、人口统计、天气等数据的分析和可视化处理，挖掘出了各部分数据的内在特征。第二部分的研究则侧重于探索出行数据与其他数据的关联。由此，我们发现了一些较为显著的特征：
- 费城的人口普查街区（census tracts）在一定程度上呈现出一定集聚性，及在人口统计上具有相似特征的街区在空间分布上彼此邻近
- 费城共享单车 Indego 的使用在工作日的通勤高峰时期最为频繁
- 费城共享单车 Indego 的使用频率明显受到天气的影响
- 费城市中心 Indego 站点的使用更为频繁
