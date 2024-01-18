import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
pd.set_option('display.max_rows'   , 1000 )
pd.set_option('display.max_columns', 1000 )
np.seterr(divide='print', over='warn', under='warn', invalid='print')
import os  # For interacting with the operating system
import seaborn as sns  # For creating statistical visualizations
import json  # For working with JSON data
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sb  # Another library for visualizations
from heatmap import heatmap  # For creating heatmaps
import plotly.express as px  # For creating interactive visualizations
import pprint  # For printing data structures in a more readable format
pp = pprint.PrettyPrinter(indent=4)
world_json = json.load(open("countries.geo.json","r"))
df_country = pd.read_csv('EdStatsCountry.csv')


df_country = df_country[['Country Code', 'Short Name', 'Long Name', 'Region' , 'Income Group' ]]
country_code_list = list(df_country['Country Code'])
macro_geographic = [ "WLD","OED","EMU","EUU","ECA","ECS","ARB","MEA","MNA","SSA","SSF","SAS","EAP","EAS" ]
macro_economic   = [ "HIC","UMC","MIC","LIC","LMC","LMY","HPC","LDC" ]

europe15 = [ "RUS","AUT","BEL","FIN","FRA","DEU","GRC","ITA","NLD","PRT","ESP","DNK","NOR","POL","SWE","CHE","GBR" ]
BRICS    = [ 'BRA','RUS','IND','CHN','ZAF' ]

country_code_list_shorter = ["AFG","ARB","ARE","ARG","AUS","AUT","BEL","BGD","BHR","BRA","CAF","ZMB","ZWE",
                             "CAN","CHE","CHL","CHN","COL","CUB","DEU","DNK","EGY","EMU","ESP","ETH","EUU",
                             "FIN","FRA","GBR","GRC","GRL","HIC","HUN","IDN","IND","IRL","IRN","IRQ","ISL",
                             "ISR","ITA","JAM","JOR","JPN","KOR","KWT","LIE","LKA","MAR","MDG","MEX","MKD",
                             "MYS","NAC","NLD","NPL","NZL","OMN","PAK","PAN","PHL","POL","PRK","PRT","QAT",
                             "RUS","SAU","SGP","SWE","THA","TUR","UKR","USA","VEN","VNM","WLD","YEM","ZAF"]
def get_country_name(country_code):
    # Assuming df_country is a DataFrame containing country information
    # and 'Short Name' and 'Country Code' are columns in the DataFrame

    # Filter the DataFrame to find the 'Short Name' where 'Country Code' matches the input country_code
    matching_names = df_country['Short Name'][df_country['Country Code'] == country_code]

    # Extract the first (and presumably only) value from the resulting Series
    country_name = matching_names.values[0]

    # Return the country name
    return country_name

# Mapping of geographic codes to corresponding geographic regions
geographic_code_name_index = {
    "EAS": "East Asia & Pacific",
    "ECS": "Europe & Central Asia",
    "LCN": "Latin America & Caribbean",
    "MEA": "Middle East & North Africa",
    "NAC": "North America",
    "SAS": "South Asia",
    "SSF": "Sub-Saharan Africa"
}

# Mapping of economic codes to corresponding income levels
economic_code_name_index = {
    "HIC": "High income: OECD",
    "HICN": "High income: nonOECD",
    "LIC": "Low income",
    "LMC": "Lower middle income",
    "UMC": "Upper middle income"
}
def get_countries_in_subset(geographic_code, economic_code):
    # Retrieve the geographic and economic names corresponding to the given codes
    geographic_name = geographic_code_name_index[geographic_code]
    economic_name = economic_code_name_index[economic_code]

    # Create a subset of country codes based on the specified geographic region
    geographic_subset = set(df_country[df_country['Region'] == geographic_name]['Country Code'].values)

    # Create a subset of country codes based on the specified income group
    economic_subset = set(df_country[df_country['Income Group'] == economic_name]['Country Code'].values)

    # Return the list of country codes that belong to both the geographic and economic subsets
    return list(geographic_subset & economic_subset)
country_count = []
for geographic_grp in geographic_code_name_index.keys() :
    for economic_grp in economic_code_name_index.keys() :
        country_count.append([ geographic_code_name_index[geographic_grp] , economic_code_name_index[economic_grp] , len(get_countries_in_subset( geographic_code = geographic_grp , economic_code = economic_grp )) ])
        
country_count = pd.DataFrame( country_count , columns = [ 'Geographic Group' , 'Economic Group' , 'Country Count'] )
country_count_pivot = country_count.pivot(index = 'Geographic Group', columns = 'Economic Group', values = 'Country Count' )
plt.figure(figsize=(8,6))

heatmap(country_count['Geographic Group'] , 
        country_count['Economic Group']   , 
        size = country_count['Country Count'] ,
        color = country_count['Country Count'] ,
        # cmap=sb.diverging_palette(20, 220 , n=200),
        palette=sb.cubehelix_palette(64)[::] , # We'll use black->red palette
        marker ='o' )
plt.show()
df_series = pd.read_csv('EdStatsSeries.csv')
df_series.Topic.value_counts()[:9].sort_values().plot.barh( figsize = (12,8) , rot = 0, 
                                                           title = 'Number of indicators availble for each topic' , 
                                                           color='darkred', fontsize = 12,
                                                           ylabel = 'Number of indicators avaible in data set for given Topic',
                                                           xlabel = 'Various Topics inside the dataset')
plt.show()
def get_indicator_code_details ( indicator_code ):
    """
    DOCSTRING :
    This funciton takes the Indicator Code as a String , Example 'SP.POP.TOTL'
    And returns a Dcitionary containing important information 
    like Full name and Defination as Key-Value Pairs

    This funcitonis useful for getting details so that 
    it can be used for generating headings and titles in graphs 
    for plots of these indicators

    """
    indicator_dict = df_series[ df_series['Series Code'] == indicator_code ].to_dict( orient = 'list' )
    
    indicator_dict = { key:val for key, val in indicator_dict.items() if type(val[0]) == str }

    return indicator_dict
df_main_data = pd.read_csv('EdStatsData.csv' )
world_df = df_main_data.copy()
df_main_data.drop(['Country Name' , 'Indicator Name' , 'Unnamed: 69'], axis = 1, inplace = True) 
#df_main_data.head()

df_main_data.drop(["2020","2025","2030","2035","2040","2045","2050","2055","2060",
                   "2065","2070","2075","2080","2085","2090","2095","2100",
                   "2016","2017"], axis = 1, inplace = True) 

df_main_data = df_main_data.dropna(thresh = 3)
year_columns = list( set( list(df_main_data.columns.values)) - set([ 'Country Code' , 'Indicator Code' ])  )
year_columns = sorted(year_columns)
data_availability = pd.DataFrame(df_main_data.count()[2:] / len(df_main_data) )[2:] * 100
data_availability.plot.bar(figsize=(14,8) , rot = 45 , title = 'Data availability for each year' , 
                    color='darkred', fontsize = 12,
                    ylabel = 'Percentage of non-Null unique values in each row',
                    xlabel = 'Years')
plt.show()
def predict(coun):
    # Read the CSV file into a DataFrame
    d = pd.read_csv('EdStatsData.csv')

    # Filter the DataFrame to include only the desired indicator
    d = d[d['Indicator Name'] == 'Adjusted net intake rate to Grade 1 of primary education, both sexes (%)']

    # Set the index of the DataFrame to 'Country Name'
    d = d.set_index('Country Name', drop=True)

    # Select columns from '1970' to '2100'
    d = d.loc[:, '1970':'2100']

    # Extract years as a list of integers
    Years = [int(year) for year in d.columns]

    # Transpose the DataFrame for easier plotting
    d_T = d.transpose()

    # Extract data for the specified country
    country_data = np.array(d_T[coun].to_list())

    # Plot a regression line for the country's data over the years
    sns.regplot(x=Years, y=country_data, line_kws={"color": "green"})
    
    # Set plot title, x-axis label, and y-axis label
    plt.title(f"Trend of intake rate to Grade 1 of primary education for {coun}")
    plt.xlabel("Years")
    plt.ylabel("Rate")

    # Display the plot
    plt.show()
def world_map_plot(indicator_code, time_period):
    # Create a copy of the world DataFrame
    tempp_df = world_df.copy()

    # Drop unnecessary columns
    tempp_df.drop(['Indicator Name', 'Unnamed: 69', "2020", "2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060",
                   "2065", "2070", "2075", "2080", "2085", "2090", "2095", "2100",
                   "2016", "2017"], axis=1, inplace=True)

    # Drop rows with insufficient data
    tempp_df = tempp_df.dropna(thresh=3)

    # Compute the mean for each five-year period
    tempp_df['1970-1975'] = tempp_df[['1970', '1971', '1972', '1973', '1974', '1975']].mean(axis=1)
    tempp_df['1976-1980'] = tempp_df[['1976', '1977', '1978', '1979', '1980']].mean(axis=1)
    tempp_df['1981-1985'] = tempp_df[['1981', '1982', '1983', '1984', '1985']].mean(axis=1)
    tempp_df['1986-1990'] = tempp_df[['1986', '1987', '1988', '1989', '1990']].mean(axis=1)
    tempp_df['1991-1995'] = tempp_df[['1991', '1992', '1993', '1994', '1995']].mean(axis=1)
    tempp_df['1996-2000'] = tempp_df[['1996', '1997', '1998', '1999', '2000']].mean(axis=1)
    tempp_df['2001-2005'] = tempp_df[['2001', '2002', '2003', '2004', '2005']].mean(axis=1)
    tempp_df['2006-2010'] = tempp_df[['2006', '2007', '2008', '2009', '2010']].mean(axis=1)
    tempp_df['2011-2015'] = tempp_df[['2011', '2012', '2013', '2014', '2015']].mean(axis=1)

    # Get the indicator name based on the indicator code
    indicator_name = get_indicator_code_details(indicator_code)['Indicator Name'][0]

    # Filter the DataFrame for the specific indicator code
    country_indi_code_df = tempp_df[tempp_df['Indicator Code'] == indicator_code]

    # Calculate the log scale for the specified time period
    country_indi_code_df['Scale'] = np.log10(country_indi_code_df[time_period])

    # Create a choropleth map using Plotly Express
    fig = px.choropleth_mapbox(country_indi_code_df,
                               geojson=world_json,
                               locations='Country Code',
                               color='Scale',
                               hover_name='Country Name',
                               hover_data=[time_period],
                               color_continuous_scale=px.colors.diverging.BrBG,
                               mapbox_style="open-street-map",
                               zoom=1,
                               opacity=0.5,
                               title=f"Plot for {indicator_name} for the time period {time_period}")
    
    # Set the height of the plot
    fig.update_layout(height=800)
    
    # Show the plot
    fig.show()

predict('Japan')
predict('Arab World')

world_map_plot('UIS.FOSEP.56.F400','2011-2015')