#!/usr/bin/env python
# coding: utf-8

# # PART I<br>DATA WRANGLING & EXPLANATORY ANALYSIS

# # 1 Demographic Variables of Census Tracts

# ## 1.1 Data Collection Through API

# In[1]:


from census_area import Census


# In[2]:


# Initialize API key
api_key = "56ee867dc43e9c68de842ea51d8b52130c9ea382"
c = Census(key=api_key)


# In[3]:


# Set FIPS codes of PA and Philly
PA_code = 42
Philly_code = 60000


# In[8]:


# Codes for demographics of interest 
codes = ['NAME','B01001_002E', 'B01001_026E','B02001_002E', 'B01002_001E', 'B06001_002E', 'B19001_001E', 'B06009_005E']
# Request ACS data
tracts = c.acs5.state_place_tract(codes, PA_code, Philly_code,
                                  return_geometry=True)


# ## 1.2 Data Wrangling

# ### Convert ot GeoDataFrame

# In[10]:


import geopandas as gpd
import pandas as pd


# In[13]:


# Initialize geographical reference system as WGS 1984 
crs = {'init': 'epsg:4269'}
tracts_gdf = gpd.GeoDataFrame.from_features(tracts, crs = crs)


# ### Rename

# In[14]:


# Rename columns of demographic variables based on correspondingly
tracts_gdf.rename(columns={'B01001_002E': 'male', 'B01001_026E': 'female',
                           'B02001_002E': 'white', 'B01002_001E': 'median_age',
                           'B06001_002E': 'under_5_years', 'B19001_001E': 'median_household_income',
                           'B06009_005E': 'bachelor'},
                  inplace=True)


# ### Select Needed Columns

# In[15]:


tracts_gdf.columns


# In[16]:


selection = ['median_age', 'median_household_income', 'population', 'pct_male', 'pct_white', 'pct_under_5_years', 'pct_bachelor', 'income_level',
            'NAME', 'GEOID', 'geometry']
tracts_gdf_selected = tracts_gdf[selection]
tracts_gdf_selected.head()


# ### Calculate the Percentages
# Since census tracts have totally different areas and population, we need to convert these extensive values (in terms of actual data totals) into intensive values (in terms of densities) by calculating each percentage.

# In[17]:


# Convert population-measured variables to the form of percentage
tracts_gdf['population'] = tracts_gdf['male'] + tracts_gdf['female'] 
tracts_gdf['pct_male'] = tracts_gdf['male'] / tracts_gdf['population'] * 100
tracts_gdf['pct_white'] = tracts_gdf['white'] / tracts_gdf['population'] * 100
tracts_gdf['pct_under_5_years'] = tracts_gdf['under_5_years'] / tracts_gdf['population'] * 100
tracts_gdf['pct_bachelor'] = tracts_gdf['bachelor'] / tracts_gdf['population'] * 100


# ### Reclassify Tracts Based on Income

# In[18]:


# Calculate the respective income of 1/3 & 2/3 quantiles
print('The 1/3 quantile of median household income is %d' %tracts_gdf['median_household_income'].quantile(0.333))
print('The 2/3 quantile of median household income is %d' %tracts_gdf['median_household_income'].quantile(0.667))


# In[19]:


# Reclassify tracts into 3 groups based on income level
tracts_gdf.loc[tracts_gdf['median_household_income'] <= 1267, 'income_level'] = 'low' 
tracts_gdf.loc[(tracts_gdf['median_household_income'] > 1267) & (tracts_gdf['median_household_income'] < 1829),  'income_level'] = 'median'
tracts_gdf.loc[tracts_gdf['median_household_income'] >= 1829, 'income_level'] = 'high'


# ## 1.3 Explanatory Analysis

# ### Numerical Relationship

# In[24]:


import hvplot.pandas


# In[28]:


# Multi-variable matrix
var_demographic = ['population', 'pct_male', 'pct_white', 'pct_under_5_years', 'pct_bachelor', 'median_household_income', 'median_age', 'income_level']

hvplot.scatter_matrix(tracts_gdf_selected[var_demographic].dropna(), c="income_level")


# In[33]:


# Correlation matrix
var_demographic2 = ['population', 'pct_male', 'pct_white', 'pct_under_5_years', 'pct_bachelor', 'median_household_income', 'median_age']

corr = tracts_gdf_selected[var_demographic2].corr()
corr.style.background_gradient(cmap='coolwarm')


# It turns out that racial composition (pct_white) is substantially related to education level (pct_bachelor). It's surprisingly that mdeian household income is statistically associated with population, that is, tracts with larger number of population exhibit higher income level.

# ### Spatial Patterns

# In[34]:


import altair as alt
alt.renderers.enable('notebook')


# In[44]:


tracts_alt = alt.InlineData(values = tracts_gdf_selected.dropna().to_json(),
                       format = alt.DataFormat(property = 'features', type = 'json'))

alt.Chart(tracts_alt).mark_geoshape(
    stroke='population',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.population:Q', 'properties.NAME:N'],
    color='properties.population:Q'
)


# In[45]:


alt.Chart(tracts_alt).mark_geoshape(
    stroke='pct_male',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.pct_male:Q', 'properties.NAME:N'],
    color='properties.pct_male:Q'
)


# In[46]:


alt.Chart(tracts_alt).mark_geoshape(
    stroke='pct_white',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.pct_white:Q', 'properties.NAME:N'],
    color='properties.pct_white:Q'
)


# In[47]:


alt.Chart(tracts_alt).mark_geoshape(
    stroke='pct_under_5_years',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.pct_under_5_years:Q', 'properties.NAME:N'],
    color='properties.pct_under_5_years:Q'
)


# In[51]:


alt.Chart(tracts_alt).mark_geoshape(
    stroke='pct_bachelor',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.pct_bachelor:Q', 'properties.NAME:N'],
    color='properties.pct_bachelor:Q'
)


# In[49]:


alt.Chart(tracts_alt).mark_geoshape(
    stroke='median_household_income',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.median_household_income:Q', 'properties.NAME:N'],
    color='properties.median_household_income:Q'
)


# In[50]:


alt.Chart(tracts_alt).mark_geoshape(
    stroke='median_age',
).properties(
    width=500,
    height=400,
    projection={"type":'mercator'},
).encode(
    tooltip=['properties.median_age:Q', 'properties.NAME:N'],
    color='properties.median_age:Q'
)


# The spatial distribution patterns of each demographic variable are presented as above respectively, which deliver a general description of
# the relationship in demographics between tracts. It's observed that census tracts in Philadelphia exhibit some degree of clustering visually, that is, census tracts with similar demographics are close to each other in space. Next, it's of more interest to look at the overall situation (similarity and difference) of these tracts, by classifying them into specific groups.

# ## 1.4 K-Means Clustering

# ### Select Variables for Clustering

# In[52]:


from sklearn.cluster import KMeans


# In[53]:


# Set the cluster number as 3
kmeans = KMeans(n_clusters = 3)


# In[54]:


var_cluster = ['median_age', 'median_household_income', 'pct_male', 'pct_white', 'pct_under_5_years', 'pct_bachelor',
                'GEOID']
tracts_cluster = tracts_gdf_selected[var_cluster].dropna()
tracts_cluster.head()


# ### Standardize Data
# For the purpose of unit scale and remove the influence of measurement units.

# In[55]:


from sklearn.preprocessing import StandardScaler


# In[56]:


# Scale data to standardized form (GEOID is excluded)
scaler = StandardScaler()
tracts_cluster_scaled = scaler.fit_transform(tracts_cluster.drop('GEOID', axis=1))

# Check whether scale-function runs successfully
tracts_cluster_scaled.std(axis = 0)


# ### Classify

# In[57]:


kmeans.fit(tracts_cluster_scaled)


# In[58]:


tracts_cluster['label'] = kmeans.labels_
tracts_cluster.head()


# In[59]:


print('The number of classified tracts using K-means is %d.' %len(tracts_cluster))


# ### Plot: Clustering over Space

# In[60]:


import pandas as pd


# In[61]:


# Join clustering labels back to GeoDataFrame tracts data
tracts_gdf_selected = pd.merge(tracts_gdf_selected, tracts_cluster[['GEOID', 'label']], left_on = 'GEOID', right_on = 'GEOID')

tracts_gdf_selected.head()


# In[64]:


from matplotlib import pyplot as plt
from collections import OrderedDict
cmaps = OrderedDict()


# In[66]:


fig, (ax1, ax2) = plt.subplots(1, 2)

tracts_gdf_selected.to_crs(epsg = '3857').plot(column = 'label', cmap = 'viridis', categorical = True, legend = True, ax = ax1)
tracts_gdf_selected.to_crs(epsg = '3857').plot(column = 'income_level', cmap = 'magma', categorical = True, legend = True, ax = ax2)

fig.set_size_inches((20,10))
ax1.set_title('K-means Clustering', fontsize = 20)
ax1.set_axis_off()
ax2.set_title('Classified by Median Household Income', fontsize = 20)
ax2.set_axis_off()


# 377 census tracts in Philadelphia are grouped into 5 classes based on demographic variables using K-means clustering. The results show that tracts locating at the central, north and southwest Philadelphia are considered more similar to each other and labeled with "2". Accordingly, the northeast, sourth, and northwest Philadelphia are classified into the same group "1". As for the group "0", there is merely no obvious distribution pattern among these tracts.
# 
# For the purpose of comparison, the spatial distribution of tracts classified by income level is also presented on the right. It turns out that tracts with relatively low median household income to some extent aggre with group "1", while the other two groups of median and high income levels are more dispersed.

# # 2 Indego Trip Data
# The Indego trip data is collected from web: https://www.rideindego.com/about/data/
# ### Import Data

# In[67]:


import pandas as pd


# In[68]:


q1 = pd.read_csv("data/indego-trips-2018-q1.csv")
q2 = pd.read_csv("data/indego-trips-2018-q2.csv")
q3 = pd.read_csv("data/indego-trips-2018-q3.csv")
q4 = pd.read_csv("data/indego-trips-2018-q4.csv")


# In[69]:


frames = [q1, q2, q3, q4]
indego = pd.concat(frames)


# ## 2.1 Temporal Patterns

# In[70]:


# Convert time-related columns to date type
indego.start_time = pd.to_datetime(indego.start_time)
indego.end_time = pd.to_datetime(indego.end_time)

# Create new columns month, hour, and day of week based on start_time column
indego['month'] = pd.DatetimeIndex(indego['start_time']).month
indego['week'] = pd.DatetimeIndex(indego['start_time']).week
indego['dow'] = indego['start_time'].dt.day_name()
indego['date'] = pd.DatetimeIndex(indego['start_time']).date
indego['hour'] = pd.DatetimeIndex(indego['start_time']).hour


# In[71]:


indego.head()


# ###  Trip Counts by Hour

# In[72]:


import seaborn as sns


# In[73]:


indego_by_hour = indego.groupby('hour').size().reset_index(name='counts')
sns.lineplot(x = 'hour', y = 'counts', data = indego_by_hour)


# Indego trips are most likely to happen during at nightfall, followed by the morning at 8:00 a.m., which corresponds well to commnon commuting schedule.

# ###  Trip Counts by Month

# In[74]:


sns.set(style="darkgrid")
sns.distplot(indego['month'], kde = False, color="m", bins = 12)


# Indego trips reach the peak during summer.

# ###  Trip Counts by Day of Week

# In[75]:


indego_by_dow = indego.groupby('dow').size().reset_index(name='counts')

dow_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

sns.barplot(x = indego_by_dow['dow'], y = indego_by_dow['counts'], palette = 'rocket', order = dow_order)


# The number of Indego trips are smaller during weekends than weekdays.

# ###  Trip Counts by Day of Week & Hour

# In[152]:


# Count trips
indego_by_dow_hour = indego.groupby(['dow', 'hour']).size().reset_index(name = 'counts')

# Sort day of week by correct order, otherwise it were ordered by first letter of dow (string) alphabetically
indego_by_dow_hour.dow = pd.Categorical(indego_by_dow_hour.dow, dow_order)
indego_by_dow_hour = indego_by_dow_hour.sort_values('dow')

indego_by_dow_hour.head()


# In[153]:


indego_by_dow_hour.hvplot.heatmap(y="dow", x="hour", 
                               C='counts', cmap='magma',
                               width=800, height=500, flip_yaxis=True)


# Indego trips in 2018 concentrate at around 8:00 a.m. and 17:00 p.m. on weekdays, which correspons well with daily commuting schedule.

# ## 2.2 Spatial Distribution

# ### Import Station Data 

# In[154]:


import json
import geopandas as gpd


# In[155]:


with open("data/stations.json") as f:
    stations_json = json.load(f)
stations = gpd.GeoDataFrame.from_features(stations_json)


# In[156]:


stations.head()


# In[157]:


stations.columns


# ### Convert to GeoDataFrame

# In[160]:


from geopandas import GeoDataFrame
from shapely.geometry import Point


# In[161]:


# Initialize the geographic reference system as WGS 1984
crs = {'init': 'epsg:4326'}
# Project ot Mercator system
stations_gdf = GeoDataFrame(stations, crs = crs, geometry = 'geometry').to_crs(epsg = 3857)
stations_gdf.head()


# ### Join Trip Counts to Station Data

# In[162]:


# Count trips by station
by_startStation = indego.groupby('start_station').size().reset_index(name = 'counts')
by_startStation.head()


# In[163]:


stations_gdf = stations_gdf.merge(by_startStation, left_on = 'kioskId', right_on = 'start_station')
stations_gdf.head()


# In[164]:


stations_gdf.columns


# ### Plot: Trip Counts at Each Station

# In[166]:


import branca.colormap as cm
color_ramp = cm.linear.YlOrRd_09.scale(0, 16000)
color_ramp


# In[167]:


import folium


# In[168]:


def get_style(feature):
    return {'weight': 2, 'color': 'white'}

def get_highlighted_style(feature):
    return {'weight': 2, 'color': 'red'}


# In[173]:


# Center the map on Philadelphia
m = folium.Map(
    location=[39.99, -75.13],
    tiles='Cartodb dark_matter',
    zoom_start=11
)

folium.GeoJson(
    tracts_gdf_selected.to_json(),
    name='Philadelphia Tracts',
    style_function=get_style,
    highlight_function=get_highlighted_style,
    tooltip=folium.GeoJsonTooltip(['NAME'])
).add_to(m)


# In[174]:


# Extract the latitude, longitude and trip counts
points = stations_gdf[['latitude', 'longitude', 'counts']].values


# In[177]:


for lat, lon, counts in points:
    folium.Circle(
        radius = 1,
        location = [lat, lon],
        color = color_ramp(counts)
    ).add_to(m)

m.save('trip counts.html')

m


# In[176]:


# ax = tracts_gdf_selected.to_crs(epsg = 3857).plot(facecolor='none', edgecolor='black')
# stations_gdf.plot(ax = ax, legend = True, markersize = 25,
#                   column = 'counts')
# ax.figure.set_size_inches((15,16))
# ax.set_title('Trip Counts in 2018 by Station', fontsize=16)


# Indego stations are distributed around center city and outer zones. Trip counts at central Philly are higher than those at outer zones, which is assumed to be a result of that tracts at center city have denser population than other areas.

# # 3 Weather
# The weather data is collected from web: https://www.wunderground.com/weather/us/pa/philadelphia

# ## 3.1 Temperature & Precipation

# ### Import Data

# In[178]:


weather = pd.read_csv("data/weather.csv")


# In[179]:


# Convert time-related columns to date type
weather.Date = pd.to_datetime(weather.Date, format='%Y%m%d')


# In[180]:


weather.head()


# ### Plot: Weather

# In[181]:


import matplotlib.pyplot as plt


# In[182]:


fig, ax1 = plt.subplots(figsize=(10, 7))

ax2 = ax1.twinx()

ax1.plot(weather['Date'], weather['Temperature Avg'], 'r-')
ax2.plot(weather['Date'], weather['Precipitation Avg'], 'b-')

# ax1.set_xlabel('Date')
ax1.set_ylabel('Average Temperature', color = 'r', fontsize = 20)
ax2.set_ylabel('Average Precipitation', color = 'b', fontsize = 20)

plt.show()


# An overview of precipitation and temperature situation of Philadelphia is presented as above. Next, we will continue to examine its relation with trips.

# ## 3.2 Indego Trips vs. Weather
# This time we compare Indego trips with average temperature & precipitation by WEEK.

# ### Wrangle Each Dataset by Week

# In[183]:


indego_by_week = indego.groupby('week').size().reset_index(name='counts')

weather['week'] = weather['Date'].dt.week
weather_by_week = weather.groupby('week').mean().reset_index()


# ### Plot: Trip Counts vs. Weather

# In[184]:


fig, (ax_left1, ax_right1) = plt.subplots(1, 2, figsize=(20, 7))

# Left: trips vs. average tempereature
ax_left2 = ax_left1.twinx()
ax_left1.plot(weather_by_week['week'], weather_by_week['Temperature Avg'], 'r-')
ax_left2.plot(indego_by_week['week'], indego_by_week['counts'], 'b-')

ax_left1.set_xlabel('Week of Year')

ax_left1.set_ylabel('Average Temperature', color='r')
ax_left2.set_ylabel('Trip Counts', color='b')

# Right: trips vs. average pricipitation
ax_right2 = ax_right1.twinx()
ax_right1.plot(weather_by_week['week'], weather_by_week['Precipitation Avg'], 'r-')
ax_right2.plot(indego_by_week['week'], indego_by_week['counts'], 'b-')

ax_right1.set_xlabel('Week of Year')
ax_right1.set_ylabel('Average Precipitation', color='r')
ax_right2.set_ylabel('Trip Counts', color='b')

plt.show()


# Temperature is more correlated to trip counts than precipitation visually. More specifically, higher temperatures agree with higher trip counts.

# # 4 Environmental Factor: Liquor Violations
# ## 4.1 Request Data

# In[216]:


import numpy as np
import cartopy.crs as ccrs
from pyrestcli.auth import NoAuthClient
from carto.sql import SQLClient


# In[217]:


sql_client = SQLClient(NoAuthClient("https://phl.carto.com"))


# In[218]:


table_name = "li_violations"
query = "SELECT COUNT(*) FROM %s" %table_name
response = sql_client.send(query)
response


# In[293]:


features = sql_client.send("SELECT * FROM %s" % table_name, format = 'geojson')
violations_gdf = gpd.GeoDataFrame.from_features(features, crs={'init':'epsg:4326'})


# In[220]:


print('The number of liquor violations is', len(violations_gdf))


# Since this dataset contains more 1 million rows, it should be ploted using datashader.

# ## 4.2 Plot: Liquor Violations Map (Large Data)

# In[294]:


selected_columns = ['geocode_x', 'geocode_y', 'geometry', 'violationdate']
violations_gdf = violations_gdf[selected_columns]

violations_gdf['year'] = violations_gdf['violationdate'].str.slice(stop=4)
violations_gdf['year'] = pd.to_numeric(violations_gdf['year'], errors='coerce')

violations_gdf.head()


# In[295]:


# Figure out the boundary of data
print("max x:", violations_gdf['geocode_x'].max())
print("min x:", violations_gdf['geocode_x'].min())
print("max y:", violations_gdf['geocode_y'].max())
print("min y:", violations_gdf['geocode_y'].min())


# In[224]:


Philly = x_range, y_range = ((2660000,2750000),(208000,305000))


# In[229]:


import holoviews as hv
import geoviews as gv
from holoviews.operation.datashader import datashade
hv.extension("bokeh")

import datashader as ds
import datashader.transfer_functions as tf

from datashader.colors import viridis
from colorcet import fire


# In[296]:


points = hv.Points(violations_gdf, kdims=['geocode_x', 'geocode_y'])


# In[297]:


plot_width  = 500
plot_height = 550


# In[298]:


datashade(points, cmap=viridis).opts(width=500, height=550, bgcolor="black")


# # PART II<br>REGRESSION ANALYSIS

# # 5 Combine All Datasets

# ## 5.1 Trip Counts by Spatial-Temporal Variables

# In[185]:


start_gp = indego.groupby(['start_station', 'month', 'week', 'dow', 'hour', 'date']).size().reset_index(name = 'counts')

# Convert date to the form of string for subsequent merging 
start_gp.date = start_gp.date.astype(str)
start_gp.head()


# It turns out that trips have been grouped by specific temporal variables and counted sucessfully.

# ## 5.2 Join Weather Data to Trip Counts (by Date)

# In[186]:


# Convert date to the form of string for subsequent merging 
weather.Date = weather.Date.astype(str)
weather.head()


# In[187]:


temp_data1 = pd.merge(start_gp, weather[['Date', 'Temperature Avg', 'Precipitation Avg']], left_on = 'date', right_on = 'Date').drop('Date', axis = 1)
temp_data1.head()


# Trip counts by station (spatial) and time-related variables (temporal) have been joined with weather data.

# ## 5.3 Spatial-Join Demographics to Trip Counts (by Location)

# In[188]:


# Merge the data from step 1.2 back to stations data to acquire the geometry info
temp_data2 = pd.merge(stations_gdf[['kioskId', 'geometry']], temp_data1, left_on = 'kioskId', right_on = 'start_station')
temp_data2.head()


# In[308]:


# Spatial-join to census tracts to acquire demographic data
temp_data3 = gpd.sjoin(temp_data2.to_crs(epsg = 3857), tracts_gdf_selected.to_crs(epsg = 3857), op = 'within', how = 'left').dropna()
temp_data3.head()


# ## 5.4 Spatial-Join Liquor Violations to Trip Counts (by Location)

# In[299]:


violations_gdf_2018 = violations_gdf[violations_gdf['year'] == 2018].dropna().reset_index()
violations_gdf_2018.head()


# In[ ]:


violations_gdf_2018_tract = gpd.sjoin(violations_gdf_2018.to_crs(epsg = 3857), tracts_gdf_selected.to_crs(epsg = 3857), op = 'within', how = 'left')


# In[306]:


violations_2018_by_tract = violations_gdf_2018_tract.groupby('GEOID').size().reset_index(name = 'liquor_violations')
violations_2018_by_tract.head()


# In[309]:


temp_data4 = pd.merge(temp_data3, violations_2018_by_tract, left_on = 'GEOID', right_on = 'GEOID')
temp_data4.head()


# Here, the final dataset including trip count, liquor violations, weather, temporal and demographic data for regression analysis has been created already.

# # 6 OLS Regression 
# ## 6.1 Modeling

# In[190]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# The statsmodels.formula.api package is used in R-style language that regresses Y on X through key variable in the form of string like "Y ~ x1 + x2 + ... +xn", so that columns named "Temperature Avg" and "Precipation Avg" with spaces are not allowed, and should be added with underline.

# In[314]:


temp_data4.rename(columns={'Temperature Avg': 'Temperature_Avg', 'Precipitation Avg': 'Precipitation_Avg'},
                  inplace=True)
reg_data = temp_data4


# In[315]:


ols = smf.ols(formula = 'counts ~ C(start_station) + C(month) + C(week) + C(dow) + C(hour) + population + Temperature_Avg + Precipitation_Avg + median_household_income + median_age + population + pct_male + pct_white + pct_under_5_years + pct_bachelor + liquor_violations', data = reg_data)

ols_results = ols.fit()


# ## 6.2 Goodness of Fitting
# ### Regression Results Summary

# In[316]:


print(ols_results.summary())


# It truns out that most of the predictors are statistically significant except for median household income indicated by p-values. Such findings are opposite to our intuitions and priori knowledge. But since the main purpose of this project is to gather, store, and analyze datasets using visualization techniques, the regression analysis serves only as an extension, so that I don't attempt to spend too much time improving this model.

# ### Predictions vs. Observations

# In[317]:


# Join predicted trip counts back to the data frame
reg_data['predicted'] = ols_results.predict()
reg_data['residuals'] = reg_data['counts'] - reg_data['predicted']


# In[318]:


reg_data.head()


# The plot below shos the relation between predicted trip counts (blue) and observed ones (orange).

# In[331]:


fig, ax = plt.subplots()

sns.set(font_scale = 2)

ax.figure.set_size_inches((10,8))

sns.regplot(x="counts", y="predicted", data=reg_data, ax = ax)
sns.regplot(x="counts", y="counts", scatter = False, data=reg_data, ax = ax)


# It's of interest to examine the temporal variance of trips of the most-used station during the most-used week.

# In[321]:


indego.groupby(['week', 'start_station']).size().reset_index(name = 'counts').sort_values('counts', ascending = False).head()


# In[323]:


week34_station3168 = reg_data.loc[(reg_data['start_station'] == 3168) & (reg_data['week'] == 34)][['dow', 'hour', 'counts', 'predicted']]
week34_station3168.head()


# In[324]:


week34_station3168_melt = pd.melt(week34_station3168, id_vars=['dow', 'hour'])
print(week34_station3168_melt.head())
print(week34_station3168_melt.tail())


# In[325]:


sns.set(font_scale = 5)

g = sns.FacetGrid(week34_station3168_melt, col = 'dow', col_wrap = 4, height = 15, hue = 'variable')
g.map(plt.plot, 'hour', 'value', marker = 'o')


# As I have explained above, though this simple OLS model does not perform perfectly in capturing the relation between trip counts and predictors, the primary obejctive of this project is not the accuracy but the process of data wrangling and visulization.

# ### Predictions vs. Residuals

# In[330]:


sns.residplot(reg_data['counts'], reg_data['residuals'], lowess=True, color="g")

