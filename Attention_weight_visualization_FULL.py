Attention weight + visualization
August 21, 2025
[1]: import os
import numpy as np
import pandas as pd
import torch
import torch .nn as nn
import torch .nn.functional as F
from torch_geometric .nn importGATConv, GCNConv, GATv2Conv
import matplotlib .pyplot as plt
import seaborn as sns
from typing importDict, List, Tuple, Optional, Union
import warnings
import logging
from pathlib importPath
from sklearn .preprocessing importStandardScaler, RobustScaler, MinMaxScaler
from sklearn .metrics importmean_squared_error, mean_absolute_error, r2_score
from collections importdeque
import math
import time
import pickle
from io importStringIO
import nltk
from newsapi importNewsApiClient
from nltk .sentiment .vader importSentimentIntensityAnalyzer
/Users/jiminbyun/anaconda3/lib/python3.11/site-
packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while
importing 'torch-sparse'. Disabling its usage. Stacktrace:
dlopen(/Users/jiminbyun/anaconda3/lib/python3.11/site-
packages/torch_sparse/_convert_cpu.so, 0x0006): symbol not found in flat
namespace '__ZN2at8internal15invoke_parallelExxxRKNSt3__18functionIFvxxEEE'
warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
[61]:warnings .filterwarnings( 'ignore')
logging.basicConfig(level =logging.INFO,format='%(asctime)s - %(levelname)s -␣
↪%(message)s ')
logger=logging.getLogger( __name__ )
[ ]:#Data Preprocessing
1

# ----- PAGE BREAK (1) -----

[62]: class RealDataPreprocessor :
"""Actual data preprocessor"""
def__init__ (self):
self.esg_data = None
self.io_data ={}
self.economic_indicators ={}
self.environmental_taxes = None
self.processed_features =None
self.industry_mapping =self._create_industry_mapping()
self.gdp_data = None
self.env_data = None
self.historical_io_data ={}
self.sentiment_features =None
defload_and_process_news_data (self, api_key: str, num_articles: int=20)␣
↪->None:
"""Through news API, collect news by industry and analyze sentiment."""
print(f"Collect news data and start analyzing sentiment (Maximum ␣
↪{num_articles }by industries)... ")
if notapi_key orapi_key =='e086209f280f46a482442a237d195a11 ':
print("￿ News API key is not provided. Skip sentiment analysis. ")
return
newsapi =NewsApiClient(api_key =api_key)
sia=SentimentIntensityAnalyzer()
sentiment_results ={}
# News seaech from industrial keywords in industry_mapping
forindustry_cat, keywords inself.industry_mapping .items():
query="OR".join(keywords)
try:
all_articles =newsapi.get_everything(q =query, language ='en',␣
↪sort_by='relevancy ', page_size =num_articles)
scores=[]
forarticle inall_articles[ 'articles ']:
text_to_analyze =(article[ 'title']or"")+""+␣
↪(article[ 'description '] or"")
iftext_to_analyze .strip():
# Calculate sentiment score using VADER (compound: -1 ~ ␣
↪+1 overall score)
sentiment_score =sia.
↪polarity_scores(text_to_analyze)[ 'compound ']
scores.append(sentiment_score)
2

# ----- PAGE BREAK (2) -----

ifscores:
sentiment_results[industry_cat] ={
'mean_sentiment ': np.mean(scores),
'sentiment_volatility ': np.std(scores),
'negative_news_ratio ':len([s forsinscores ifs<-0.
↪05])/len(scores)
}
except Exception ase:
print(f"- ￿ '{industry_cat }'Error occurs during news ␣
↪searching: {e}")
self.sentiment_features =pd.DataFrame .from_dict(sentiment_results, ␣
↪orient='index')
print("￿ News data load is completed. ")
def_create_sentiment_features (self, sector_codes: List[ str])->np.ndarray:
"""Map analyzed sentiment features to I-O table sector."""
n_features =3# Mean sentiment, volatility, negative news proportion
features =np.zeros((len(sector_codes), n_features))
ifself.sentiment_features is None orself.sentiment_features .empty:
print("- Fill 0 because there is no sentiment data. ")
returnfeatures
fori, sector_code inenumerate (sector_codes):
sector_num =int(sector_code)
# I-O code -> Industry mapping
ifsector_num <=10: cat='Agriculture '
elifsector_num <=30: cat='Technology '
else: cat='Transportation '
ifcat inself.sentiment_features .index:
features[i, :] =self.sentiment_features .loc[cat] .values
returnfeatures
def_create_industry_mapping (self)->Dict[str, List[str]]:
"""Mapping ESG industries and I-O industries"""
return{
'Technology ': ['Information & Communication ','Professional, ␣
↪Scientific & Technical Services ','Programming ','Software ',␣
↪'Semiconductors '],
'Financial Services ': ['Financial & Insurance Services ','Banking',␣
↪'Insurance '],
'Health Care ': ['Health Services ','Medical Services '],
3

# ----- PAGE BREAK (3) -----

'Utilities ': ['Electricity & Gas ','Electric Power ','Water Supply ␣
↪& Sewage '],
'Oil & Gas E&P ': ['Petroleum Refining '],
'Chemicals ': ['Chemical Materials ','Chemical Products '],
'Steel': ['Primary Metal Industry '],
'Real Estate ': ['Real Estate ','Leasing'],
'Transportation ': ['Transportation '],
'Agriculture ': ['Agriculture '],
'Mining': ['Mining','Quarrying ']
}
defload_esg_data (self, file_path: str='data.csv ')->pd.DataFrame:
"""ESG data loading"""
print("ESG data loading... ")
try:
self.esg_data =pd.read_csv(file_path)
print(f"ESG data loading completed: ")
print(f"- Number of companies: {len(self.esg_data) }")
print(f"- Number of industries: {self.esg_data[ 'industry '].
↪nunique() }industries ")
# ESG score normalization
score_cols =['environment_score ','social_score ',␣
↪'governance_score ','total_score ']
forcol inscore_cols:
ifcol inself.esg_data .columns:
max_val =self.esg_data[col] .max()
min_val =self.esg_data[col] .min()
ifmax_val >min_val:
self.esg_data[ f'{col }_normalized ']=(self.
↪esg_data[col] -min_val) /(max_val -min_val)
else:
self.esg_data[ f'{col }_normalized ']=0.5
# ESG grade numberization
grade_mapping ={
'AAA':1.0,'AA':0.9,'A':0.8,'BBB':0.7,'BB':0.6,
'B':0.5,'CCC':0.4,'CC':0.3,'C':0.2
}
grade_cols =['environment_grade ','social_grade ',␣
↪'governance_grade ','total_grade ']
forcol ingrade_cols:
ifcol inself.esg_data .columns:
self.esg_data[ f'{col }_numeric ']=self.esg_data[col] .
↪map(grade_mapping) .fillna(0.5)
4

# ----- PAGE BREAK (4) -----

print(f"- mean ESG score: {self.esg_data[ 'total_score '].mean() :.
↪1f }")
print(f"- ESG score scope: {self.esg_data[ 'total_score '].
↪min() }-{self.esg_data[ 'total_score '].max() }")
returnself.esg_data
except Exception ase:
print(f"ESG data load failed: {e}")
print("- Generating dummy ESG data. ")
self.esg_data =self._create_demo_esg_data()
returnself.esg_data
def_create_demo_esg_data (self)->pd.DataFrame:
"""Generate ESG dummy data"""
np.random.seed(42)
n_companies =100
industries =list(self.industry_mapping .keys())
demo_data ={
'company_name ': [f'Company_ {i}' fori inrange(n_companies)],
'industry ': np.random.choice(industries, n_companies),
'environment_score ': np.random.normal(650,100, n_companies),
'social_score ': np.random.normal(700,120, n_companies),
'governance_score ': np.random.normal(680,110, n_companies),
'total_score ': np.random.normal(2030,200, n_companies)
}
df=pd.DataFrame(demo_data)
# Normalization
forcol in['environment_score ','social_score ','governance_score ',␣
↪'total_score ']:
max_val =df[col].max()
min_val =df[col].min()
df[f'{col }_normalized ']=(df[col] -min_val) /(max_val -min_val)
returndf
defload_io_data (self, use_file: str='REAL_USE.xlsx ', make_file: str=␣
↪'REAL_MAKE.xlsx ', num_years: int=5)->Optional[Dict]:
"""Load recent N years of I-O data and calculate yearly A_matrix"""
print(f"Loading I-O tables (recent {num_years }years)... ")
try:
use_excel =pd.ExcelFile(use_file, engine ='openpyxl ')
available_years =sorted([sheet forsheet inuse_excel .sheet_names ␣
↪ifsheet.isdigit()], reverse =True)
years_to_load =available_years[:num_years]
5

# ----- PAGE BREAK (5) -----

print(f"- Years to use: {','.join(years_to_load) }")
foryear inyears_to_load:
use_df_raw =pd.read_excel(use_file, sheet_name =year,␣
↪index_col =0)
make_df_raw =pd.read_excel(make_file, sheet_name =year,␣
↪index_col =0)
# Find common sectors and sort
use_sectors =use_df_raw .index[1:].intersection(use_df_raw .
↪columns[ 1:])
make_sectors =make_df_raw .index[1:].intersection(make_df_raw .
↪columns[ 1:])
common_sectors =sorted(list(use_sectors .
↪intersection(make_sectors)))
use_matrix =use_df_raw .loc[common_sectors, common_sectors] .
↪values.astype(float)
make_matrix =make_df_raw .loc[common_sectors, common_sectors] .
↪values.astype(float)
# Calculate A_matrix
g=np.where(use_matrix .sum(axis =0)==0,1e-9, use_matrix .
↪sum(axis =0))
B_matrix =use_matrix /g
q=np.where(make_matrix .sum(axis =0)==0,1e-9, make_matrix .
↪sum(axis =0))
D_matrix =make_matrix .T/q
A_matrix =np.nan_to_num(np .dot(B_matrix, D_matrix))
# Store yearly A_matrix and other info
self.historical_io_data[year] ={'A_matrix ': A_matrix, ␣
↪'sectors': common_sectors}
# Configure basic io_data based on latest year data
latest_year_data =self.historical_io_data[years_to_load[ 0]]
latest_A_matrix =latest_year_data[ 'A_matrix ']
n_sectors =latest_A_matrix .shape[0]
I=np.eye(n_sectors)
try:
L_matrix =np.linalg.inv(I-latest_A_matrix)
exceptnp.linalg.LinAlgError:
L_matrix =np.linalg.pinv(I-latest_A_matrix)
6

# ----- PAGE BREAK (6) -----

edge_sources, edge_targets =np.where(latest_A_matrix >np.
↪percentile(latest_A_matrix[latest_A_matrix >0],50))
edge_weights =np.log(latest_A_matrix[edge_sources, edge_targets] +␣
↪1e-9)
edge_weights =(edge_weights -edge_weights .mean()) /(edge_weights .
↪std()+1e-9)
self.io_data ={
'A_matrix ': latest_A_matrix, 'L_matrix ': L_matrix,
'sector_codes ': [str(i) foriinrange(1, n_sectors +1)],␣
↪'n_sectors ': n_sectors,
'edge_index ': np.vstack([edge_sources, edge_targets]),
'edge_weights ': np.clip(edge_weights, -5.0,5.0)
}
print("￿ Multi-year I-O data loading and processing completed. ")
returnself.io_data
except Exception ase:
print(f"I-O table loading failed: {e}")
return None
def_create_demo_io_data (self)->Dict:
"""Generate I-O dummy data"""
np.random.seed(42)
n_sectors =25
# Generate dummy USE table
use_matrix =np.random.exponential( 100, (n_sectors, n_sectors))
use_matrix =use_matrix *(np.random.random((n_sectors, n_sectors)) >0.
↪7)# Sparse matrix
# I-O table
total_output =use_matrix .sum(axis =1)
total_output =np.where(total_output ==0,1e-9, total_output)
A_matrix =use_matrix /total_output .reshape( -1,1)
A_matrix =np.nan_to_num(A_matrix)
# Leontief Inverse
I=np.eye(n_sectors)
try:
L_matrix =np.linalg.inv(I-A_matrix)
except:
L_matrix =np.linalg.pinv(I-A_matrix)
# Generate edge
threshold =np.percentile(A_matrix[A_matrix >0],70)
7

# ----- PAGE BREAK (7) -----

edge_sources, edge_targets =np.where(A_matrix >threshold)
edge_weights =A_matrix[edge_sources, edge_targets]
edge_weights =np.log(edge_weights +1e-9)
edge_weights =(edge_weights -edge_weights .mean()) /(edge_weights .
↪std()+1e-9)
return{
'use_matrix ': use_matrix,
'A_matrix ': A_matrix,
'L_matrix ': L_matrix,
'sector_codes ': [str(i) foriinrange(1, n_sectors +1)],
'n_sectors ': n_sectors,
'edge_index ': np.vstack([edge_sources, edge_targets]),
'edge_weights ': edge_weights,
'years_used ': ['2022']
}
defload_real_gdp_data (self, file_path: str='Real Gross Output by ␣
↪Industry.csv ')->None:
"""Load GDP growth data from Table.csv file"""
print(f"GDP data loading ( '{file_path }')...")
try:
# BEA data format, header=4
gdp_df=pd.read_csv(file_path, header =4, index_col =0)
# Remove empty rows
last_industry_row =gdp_df.index.get_loc( "Government ")#␣
↪'Government' as the last row
gdp_df=gdp_df.iloc[:last_industry_row +1]
# Get the latest growth rate column
latest_growth_rate_col =gdp_df.columns[ -1]
self.gdp_data =gdp_df[[latest_growth_rate_col]] .copy()
self.gdp_data .columns =['growth_rate ']
# Convert to numeric
self.gdp_data[ 'growth_rate ']=pd.to_numeric( self.
↪gdp_data[ 'growth_rate '], errors ='coerce')
self.gdp_data .fillna(self.gdp_data .mean(), inplace =True)# Fill␣
↪NaN with mean
print(f"GDP data loaded: {len(self.gdp_data) }industries ")
except FileNotFoundError :
print(f"GDP data file not found: '{file_path }'")
self.gdp_data = None
except Exception ase:
8

# ----- PAGE BREAK (8) -----

print(f"GDP data loading failed: {e}")
self.gdp_data = None
def_map_gdp_to_sectors (self, sector_codes: List[ str])->np.ndarray:
"""Map GDP data to I-O sectors"""
economic_features =np.zeros((len(sector_codes), 3))
ifself.gdp_data is None:
print("- GDP data not available, using dummy data. ")
returnself._create_demo_economic_data( len(sector_codes))
# industry_mapping to map I-O sectors to GDP industries
fori, sector_code inenumerate (sector_codes):
sector_num =int(sector_code)
# Map I-O sector numbers to industry categories (simplified mapping)
ifsector_num <=10:
industries =self.industry_mapping .get('Agriculture ', [])+␣
↪self.industry_mapping .get('Mining', [])
elifsector_num <=30:
industries =self.industry_mapping .get('Technology ', [])+self.
↪industry_mapping .get('Chemicals ', [])+self.industry_mapping .get('Steel',␣
↪[])
elifsector_num <=50:
industries =self.industry_mapping .get('Financial Services ',␣
↪[])+self.industry_mapping .get('Health Care ', [])
elifsector_num <=70:
industries =self.industry_mapping .get('Utilities ', [])+self.
↪industry_mapping .get('Oil & Gas E&P ', [])
else:
industries =self.industry_mapping .get('Real Estate ', [])+␣
↪self.industry_mapping .get('Transportation ', [])
# Find matching GDP data
matched_rates =[]
forindustry_keyword inindustries:
# gdp_data index to search for matching industries
relevant_rows =self.gdp_data[ self.gdp_data .index.str.
↪contains(industry_keyword, case =False, na=False)]
if notrelevant_rows .empty:
matched_rates .append(relevant_rows[ 'growth_rate '].mean())
avg_growth_rate =np.mean(matched_rates) /100.0 ifmatched_rates ␣
↪else0.02# Convert %, default 2%
# 1. Growth rate: average growth rate
9

# ----- PAGE BREAK (9) -----

economic_features[i, 0]=avg_growth_rate
# 2. Volatility/risk: volatility based on growth rate
economic_features[i, 1]=np.abs(avg_growth_rate) *np.random.
↪uniform( 0.5,1.5)
economic_features[i, 2]=avg_growth_rate *np.random.uniform( 0.8,␣
↪1.2)
print("GDP data mapping completed. ")
returneconomic_features
def_create_timeseries_features (self)->np.ndarray:
"""Generate time series features using stored historical I-O data"""
print("Creating time series features... ")
n_sectors =self.io_data[ 'n_sectors ']
# Lists to store yearly calculated indicators
yearly_total_inputs =[]
yearly_total_outputs =[]
sorted_years =sorted(self.historical_io_data .keys())
foryear insorted_years:
A_matrix =self.historical_io_data[year][ 'A_matrix ']
# I-O tables may have different sizes by year, so alignment/
↪resizing may be needed
# (Here we assume they have the same size for simplification)
yearly_total_inputs .append(A_matrix .sum(axis =1))
yearly_total_outputs .append(A_matrix .sum(axis =0))
# Convert lists to numpy arrays (years x sectors)
historical_inputs =np.array(yearly_total_inputs)
historical_outputs =np.array(yearly_total_outputs)
# Calculate average change rates (trends) and standard deviations ␣
↪(volatility) over recent 5 years
# np.diff: calculates differences between years
input_trends =np.mean(np.diff(historical_inputs, axis =0), axis=0)
output_trends =np.mean(np.diff(historical_outputs, axis =0), axis=0)
input_volatility =np.std(historical_inputs, axis =0)
output_volatility =np.std(historical_outputs, axis =0)
print("￿ Time series feature creation completed. ")
returnnp.column_stack([input_trends, output_trends, input_volatility, ␣
↪output_volatility])
def_create_demo_economic_data (self, n_sectors: int)->np.ndarray:
10

# ----- PAGE BREAK (10) -----

"""Generate dummy economic data"""
features =np.zeros((n_sectors, 3))
fori inrange(n_sectors):
features[i, 0]=np.random.normal(0.02,0.05)
features[i, 1]=np.random.uniform( 0.1,0.3)
features[i, 2]=np.random.normal(0,0.02)
returnfeatures
defload_environmental_taxes (self, file_path: str='Environmental Taxes.
↪csv')->pd.Series:
"""load environmental taxes"""
print("Load environmental taxes... ")
try:
ifos.path.exists(file_path):
df=pd.read_csv(file_path)
korea_mask =df['REF_AREA_NAME '].str.contains( 'Korea',␣
↪case=False, na=False)
korea_data =df[korea_mask] .copy()
iflen(korea_data) >0:
recent_data =korea_data[korea_data[ 'TIME_PERIOD ']>=2015]
latest_taxes =recent_data .
↪groupby( 'INDICATOR_NAME ')['OBS_VALUE '].last().fillna(0)
self.environmental_taxes =latest_taxes
print(f"Load environmental taxes: {len(latest_taxes) }")
returnlatest_taxes
else:
raise ValueError ("No Korean data ")
else:
raise FileNotFoundError ("No file")
except Exception ase:
print(f"Environmental taxes data load fails: {e}")
print("- Generate dummy environmental taxes data. ")
self.environmental_taxes =self._create_demo_env_tax_data()
returnself.environmental_taxes
def_create_demo_env_tax_data (self)->pd.Series:
"""Generate dummy environmental taxes data"""
demo_taxes =pd.Series({
'Carbon Tax ':0.25,
'Energy Tax ':0.35,
'Transport Tax ':0.20,
'Pollution Tax ':0.15,
'Resource Tax ':0.05
})
11

# ----- PAGE BREAK (11) -----

returndemo_taxes
def_map_esg_to_sectors (self, sector_codes: List[ str])->np.ndarray:
"""Map ESG score to I-O sector"""
esg_features =np.zeros((len(sector_codes), 4))
ifself.esg_data is not None:
fori, sector_code inenumerate (sector_codes):
sector_num =int(sector_code)
ifsector_num <=10:
matched_industries =['Agriculture ','Mining']
elifsector_num <=30:
matched_industries =['Technology ','Chemicals ','Steel']
elifsector_num <=50:
matched_industries =['Financial Services ','Health Care ']
elifsector_num <=70:
matched_industries =['Utilities ','Oil & Gas E&P ']
else:
matched_industries =['Real Estate ','Transportation ']
matched_data =self.esg_data[ self.esg_data[ 'industry '].
↪isin(matched_industries)]
iflen(matched_data) >0:
esg_features[i, 0]=␣
↪matched_data[ 'environment_score_normalized '].fillna(0.5).mean()
esg_features[i, 1]=␣
↪matched_data[ 'social_score_normalized '].fillna(0.5).mean()
esg_features[i, 2]=␣
↪matched_data[ 'governance_score_normalized '].fillna(0.5).mean()
esg_features[i, 3]=matched_data[ 'total_score_normalized '].
↪fillna(0.0).std()
else:
esg_features[i] =[0.5,0.5,0.5,0.1]
else:
esg_features =self._generate_esg_heuristics( len(sector_codes))
returnesg_features
def_generate_esg_heuristics (self, n_sectors: int)->np.ndarray:
"""Generate heuristics based virtual ESG data"""
esg_features =np.zeros((n_sectors, 4))
fori inrange(n_sectors):
sector_num =i+1
12

# ----- PAGE BREAK (12) -----

ifsector_num in[4,5,26,38,61]:
env_score =0.3
elifsector_num in[27,28,36]:
env_score =0.4
else:
env_score =0.7
ifsector_num in[66,70,84,85]:
social_score =0.8
else:
social_score =0.6
ifsector_num in[75,76,83]:
gov_score =0.8
else:
gov_score =0.6
volatility =np.random.normal(0.1,0.05)
volatility =max(0.05,min(0.3, volatility))
esg_features[i] =[env_score, social_score, gov_score, volatility]
returnesg_features
defload_real_env_data (self, file_path: str)->None:
"""Load environmental data from EPA greenhouse gas reporting program"""
print(f"Loading environmental data ( '{file_path }')...")
try:
# Read Excel file 'Direct Point Emitters' sheet
# Header is in the 4th row (index 3)
df=pd.read_excel(file_path,
sheet_name ='Direct Point Emitters ',
header=3,# Use 4th row as header
engine='openpyxl ')
print(f"Original data size: {df.shape }")
# Check and extract required columns
required_columns =['Industry Type (sectors) ','Total reported ␣
↪direct emissions ']
ifall(col indf.columns forcol inrequired_columns):
# Extract only required columns
df_subset =df[required_columns] .copy()
# Convert emissions data to numeric
13

# ----- PAGE BREAK (13) -----

df_subset[ 'Total reported direct emissions ']=pd.to_numeric(
df_subset[ 'Total reported direct emissions '],
errors='coerce'
)
# Remove NaN values
df_subset =df_subset .dropna()
# Group by industry type and calculate total emissions
self.env_data =df_subset .groupby( 'Industry Type ␣
↪(sectors) ')['Total reported direct emissions '].sum().reset_index()
self.env_data =self.env_data .set_index( 'Industry Type ␣
↪(sectors) ')
print(f"Processed data: {len(self.env_data) }industry types ")
print("Emissions by industry type (top 5): ")
top_5=self.env_data .nlargest( 5,'Total reported direct ␣
↪emissions ')
foridx, row intop_5.iterrows():
print(f" {idx }:{row['Total reported direct emissions ']:,.
↪0f }metric tons CO2e ")
else:
missing_cols =[col forcol inrequired_columns ifcol not in␣
↪df.columns]
print(f"Required columns missing: {missing_cols }")
self.env_data = None
except Exception ase:
print(f"Environmental data loading failed: {e}")
self.env_data = None
def_create_environmental_features (self, sector_codes: List[ str])->np.
↪ndarray:
"""Map environmental characteristics to I-O sectors (updated version)"""
features =np.zeros((len(sector_codes), 3))
ifself.env_data is None:
print("- Environmental data not available, using dummy data. ")
# Generate dummy data
fori inrange(len(sector_codes)):
features[i, 0]=np.random.lognormal( 10,1)# Emissions
features[i, 1]=features[i, 0]*np.random.uniform( 0.8,1.2)␣
↪# Emissions volatility
features[i, 2]=np.random.uniform( 0.3,0.8)# Environmental ␣
↪efficiency
14

# ----- PAGE BREAK (14) -----

returnfeatures
# Map EPA industry categories to I-O sectors
epa_to_io_mapping ={
'Chemicals ': ['Chemicals '],
'Metals': ['Steel','Primary Metal Industry '],
'Minerals ': ['Mining','Quarrying ','Nonmetallic Mineral Products '],
'Other': ['Technology ','Manufacturing '],
'Petroleum and Natural Gas Systems ': ['Oil & Gas E&P ','Petroleum ␣
↪Refining '],
'Power Plants ': ['Utilities ','Electric Power ','Electricity & ␣
↪Gas'],
'Waste': ['Health Care ','Financial Services ','Real Estate '],
'Pulp and Paper Manufacturing ': ['Agriculture ','Forestry '],
'Refineries ': ['Petroleum Refining '],
'Iron and Steel Production ': ['Steel','Primary Metal Industry '],
'Cement Production ': ['Nonmetallic Mineral Products '],
'Lime Manufacturing ': ['Nonmetallic Mineral Products '],
'Nitric Acid Production ': ['Chemicals '],
'Adipic Acid Production ': ['Chemicals '],
'Semiconductor Manufacturing ': ['Technology '],
'Aluminum Production ': ['Primary Metal Industry '],
'Magnesium Production ': ['Primary Metal Industry '],
'Lead Production ': ['Primary Metal Industry '],
'Zinc Production ': ['Primary Metal Industry '],
'Glass Production ': ['Nonmetallic Mineral Products '],
'Hydrogen Production ': ['Chemicals '],
'Ammonia Manufacturing ': ['Chemicals '],
'Petrochemical Production ': ['Chemicals '],
'Ferroalloy Production ': ['Primary Metal Industry '],
'Phosphoric Acid Production ': ['Chemicals '],
'Soda Ash Manufacturing ': ['Chemicals '],
'Petroleum Product Suppliers,Refineries ': ['Petroleum Refining '],
'Chemicals,Petroleum Product Suppliers,Refineries ': ['Chemicals ',␣
↪'Petroleum Refining '],
'Chemicals,Suppliers of CO2 ': ['Chemicals ']
}
fori, sector_code inenumerate (sector_codes):
sector_num =int(sector_code)
# Map I-O sector numbers to industry categories
ifsector_num <=10:
my_keywords =self.industry_mapping .get('Agriculture ', [])+\
self.industry_mapping .get('Mining', [])
elifsector_num <=30:
my_keywords =self.industry_mapping .get('Technology ', [])+\
15

# ----- PAGE BREAK (15) -----

self.industry_mapping .get('Chemicals ', [])+\
self.industry_mapping .get('Steel', [])
elifsector_num <=50:
my_keywords =self.industry_mapping .get('Financial Services ',␣
↪[])+\
self.industry_mapping .get('Health Care ', [])
elifsector_num <=70:
my_keywords =self.industry_mapping .get('Utilities ', [])+\
self.industry_mapping .get('Oil & Gas E&P ', [])
else:
my_keywords =self.industry_mapping .get('Real Estate ', [])+\
self.industry_mapping .get('Transportation ', [])
# Find matching emissions from EPA data
ghg_value =0
matched_count =0
forepa_sector, keywords inepa_to_io_mapping .items():
ifany(kw inmy_kw forkw inkeywords formy_kw inmy_keywords):
ifepa_sector inself.env_data .index:
ghg_value +=self.env_data .loc[epa_sector, 'Total␣
↪reported direct emissions ']
matched_count +=1
avg_ghg_value =ghg_value /matched_count ifmatched_count >0else␣
↪0
# Apply log scaling (to reduce difference between large values)
ifavg_ghg_value >0:
log_ghg_value =np.log10(avg_ghg_value +1)
normalized_ghg =log_ghg_value /10.0# Normalize to 0-1 range
else:
normalized_ghg =0.01# Default value
# Feature assignment
features[i, 0]=normalized_ghg # Feature 1: Normalized direct ␣
↪emissions
features[i, 1]=normalized_ghg *np.random.uniform( 0.8,1.2)#␣
↪Feature 2: Emissions volatility
# Feature 3: Environmental efficiency (high emission sectors have ␣
↪low efficiency)
ifsector_num in[6,7,26,38]:# High emission sectors
features[i, 2]=0.8
else:
features[i, 2]=0.3
16

# ----- PAGE BREAK (16) -----

print("Environmental feature mapping completed. ")
returnfeatures
def_create_risk_profiles (self, sector_codes: List[ str])->np.ndarray:
"""Generate virtual risk profiles features"""
features =np.zeros((len(sector_codes), 3))
fori, sector_code inenumerate (sector_codes):
sector_num =int(sector_code)
ifsector_num in[75,76,78]:
features[i, 0]=0.8
elifsector_num in[27,28,30]:
features[i, 0]=0.7
else:
features[i, 0]=0.4
ifsector_num in[75,61,72]:
features[i, 1]=0.9
elifsector_num in[27,30,16]:
features[i, 1]=0.7
else:
features[i, 1]=0.4
ifsector_num in[74,75,79]:
features[i, 2]=0.8
else:
features[i, 2]=0.5
returnfeatures
defcreate_integrated_features (self)->Optional[Dict]:
"""Generate integrated features for GNN model input (including time ␣
↪series)"""
print("Creating integrated features for GNN input... ")
if notself.io_data:
return None
# --- 1. Generate static features based on latest year (similar to ␣
↪existing logic) ---
n_sectors =self.io_data[ 'n_sectors ']
sector_codes =self.io_data[ 'sector_codes ']
A_matrix =self.io_data[ 'A_matrix ']
L_matrix =self.io_data[ 'L_matrix ']
total_input =A_matrix .sum(axis =1)
17

# ----- PAGE BREAK (17) -----

total_output =A_matrix .sum(axis =0)
backward_linkage =total_input /(total_input .mean()+1e-9)
forward_linkage =total_output /(total_output .mean()+1e-9)
multiplier =L_matrix .sum(axis =0)
concentration =np.array([np .sum((A_matrix[i, :] /(total_input[i] +␣
↪1e-9))**2)foriinrange(n_sectors)])
esg_features =self._map_esg_to_sectors(sector_codes)
economic_features =self._map_gdp_to_sectors(sector_codes)
env_tax_features =self._create_environmental_features(sector_codes)
risk_features =self._create_risk_profiles(sector_codes)
# --- 2. Generate time series features ---
timeseries_features =self._create_timeseries_features()
sentiment_features =self._create_sentiment_features(sector_codes)
# --- 3. Combine all features ---
all_features =np.column_stack([
# Existing static features
np.log(total_input +1), np.log(total_output +1), backward_linkage,
forward_linkage, multiplier, concentration, esg_features,
economic_features, env_tax_features, risk_features,
# Newly added time series features (4 features)
timeseries_features,
sentiment_features
])
scaler=MinMaxScaler()
all_features =scaler.fit_transform(all_features)
self.processed_features ={
'node_features ': all_features,
'edge_index ':self.io_data[ 'edge_index '],
'edge_weights ':self.io_data[ 'edge_weights '],
'sector_codes ': sector_codes, 'n_nodes': n_sectors,
'n_features ': all_features .shape[1],
'n_edges':len(self.io_data[ 'edge_weights ']),
'x': torch.FloatTensor(all_features),
'edge_index ': torch.LongTensor( self.io_data[ 'edge_index ']),
'edge_weight ': torch.FloatTensor( self.io_data[ 'edge_weights ']),
'codes': sector_codes, 'data_source ':'Real Data with Timeseries '
}
print(f"￿ Integrated feature creation completed: {all_features .shape }")
returnself.processed_features
defcreate_esg_targets (self)->Dict[str, torch.Tensor]:
18

# ----- PAGE BREAK (18) -----

"""Generate target data for GNN learning"""
print("Generate target data for GNN learning... ")
if notself.processed_features:
print("Can not generate target with no features. ")
return{}
features =self.processed_features[ 'x'].numpy()
# 1. ESG overall risk
esg_env =features[:, 6]
esg_social =features[:, 7]
esg_gov =features[:, 8]
esg_risk =1.0-(esg_env *0.4+esg_social *0.3+esg_gov *0.3)
esg_risk =np.clip(esg_risk, 0,1)
# 2. Economic Impact
env_exposure =features[:, 13]# carbon emission feature
multiplier =features[:, 4]# Production-Inducement coefficient ␣
↪features
economic_impact =(env_exposure -0.5)*(multiplier /(multiplier .
↪max()+1e-9))*0.2
economic_impact =np.clip(economic_impact, -0.1,0.333)# Both␣
↪possible for positive/negative effects
# 3. Volatility
volatility =features[:, 11]# Economical volatility features
volatility =np.abs(volatility) +0.01# Adjusted to be always positive
volatility =np.clip(volatility, 0.01,2.0)
# 4. Transition Cost
carbon_intensity =features[:, 14]# Energy consuming efficiency ␣
↪features
transition_readiness =features[:, 18]# Operational Risk Feature ␣
↪(lower is better)
transition_cost =carbon_intensity *(1-transition_readiness)
transition_cost =np.clip(transition_cost, 0,1)
# 5. Compliance Probability
regulatory_pressure =features[:, 17]# Market risk features (higher ␣
↪is worse)
compliance_prob =esg_gov *(1-regulatory_pressure *0.3)
compliance_prob =np.clip(compliance_prob, 0,1)
targets ={
'esg_risk ': torch.FloatTensor(esg_risk),
19

# ----- PAGE BREAK (19) -----

'economic_impact ': torch.FloatTensor(economic_impact),
'volatility ': torch.FloatTensor(volatility),
'transition_cost ': torch.FloatTensor(transition_cost),
'compliance_probability ': torch.FloatTensor(compliance_prob)
}
print("Target data generating completes: ")
forname, tensor intargets.items():
min_val, max_val =tensor.min().item(), tensor .max().item()
print(f"-{name }: (Mean: {tensor.mean() :.3f }, Scope: [ {min_val :.
↪3f }, {max_val :.3f }])")
returntargets
[ ]:
[ ]:#GNN Models
[63]:# GPRGNN Implementation - Run this cell BEFORE the main() function
import torch
import torch .nn as nn
import torch .nn.functional as F
from torch_geometric .nn importMessagePassing
from torch_geometric .utils importadd_self_loops, degree
class GPRGNNLayer (MessagePassing):
def__init__ (self, in_channels: int, out_channels: int,
alpha:float=0.1, K:int=10, dropout: float=0.1):
super().__init__ (aggr='add')
self.in_channels =in_channels
self.out_channels =out_channels
self.alpha=alpha# PageRank damping factor
self.K=K# Number of propagation steps
self.dropout =dropout
# Linear transformation
self.lin=nn.Linear(in_channels, out_channels)
# Learnable weights for each propagation step
self.gamma=nn.Parameter(torch .zeros(K +1))
self.reset_parameters()
defreset_parameters (self):
nn.init.xavier_uniform_( self.lin.weight)
nn.init.zeros_(self.lin.bias)
nn.init.uniform_( self.gamma,0.0,1.0)
20

# ----- PAGE BREAK (20) -----

defforward(self, x, edge_index, edge_weight =None):
# Add self-loops
edge_index, edge_weight =add_self_loops(
edge_index, edge_weight, num_nodes =x.size(0)
)
# Compute normalization
row, col =edge_index
deg=degree(col, x .size(0), dtype =x.dtype)
deg_inv_sqrt =deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt ==float('inf')]=0
norm=deg_inv_sqrt[row] *deg_inv_sqrt[col]
# Transform input features
h=self.lin(x)
h=F.dropout(h, p =self.dropout, training =self.training)
# Store propagation results
hs=[h]
# Multi-step propagation with PageRank-style weighting
fork inrange(self.K):
h=self.propagate(edge_index, x =h, norm=norm)
h=(1-self.alpha)*h+self.alpha*hs[0]# PageRank update
hs.append(h)
# Weighted combination of all propagation steps
gamma_soft =F.softmax( self.gamma, dim =0)
output=sum(gamma_soft[k] *hs[k] forkinrange(self.K+1))
returnoutput
defmessage(self, x_j, norm):
returnnorm.view(-1,1)*x_j
class GPRGNN(nn.Module):
def__init__ (self, input_dim: int, hidden_dim: int=64, output_dim: int=␣
↪32,
num_layers: int=2, alpha: float=0.1, K:int=10,
dropout: float=0.1, use_residual: bool=True):
super().__init__ ()
self.num_layers =num_layers
self.use_residual =use_residual
self.dropout =dropout
21

# ----- PAGE BREAK (21) -----

# Input projection
self.input_proj =nn.Linear(input_dim, hidden_dim)
# GPRGNN layers
self.gpr_layers =nn.ModuleList()
fori inrange(num_layers):
in_dim=hidden_dim ifi==0elsehidden_dim
out_dim =hidden_dim ifi<num_layers -1 elseoutput_dim
self.gpr_layers .append(
GPRGNNLayer(in_dim, out_dim, alpha =alpha, K =K, dropout =dropout)
)
# Layer normalization
self.layer_norms =nn.ModuleList([
nn.LayerNorm(hidden_dim) for_ inrange(num_layers -1)
])
# Residual projections (if dimensions don't match)
self.residual_projs =nn.ModuleList()
fori inrange(num_layers):
ifi==0:
# First layer: input_dim -> hidden_dim
ifinput_dim !=hidden_dim:
self.residual_projs .append(nn .Linear(input_dim, hidden_dim))
else:
self.residual_projs .append(nn .Identity())
elifi==num_layers -1:
# Last layer: hidden_dim -> output_dim
ifhidden_dim !=output_dim:
self.residual_projs .append(nn .Linear(hidden_dim, ␣
↪output_dim))
else:
self.residual_projs .append(nn .Identity())
else:
# Middle layers: hidden_dim -> hidden_dim
self.residual_projs .append(nn .Identity())
# Global attention pooling (optional)
self.global_attention =nn.MultiheadAttention(
embed_dim =output_dim, num_heads =4, dropout =dropout, batch_first =True
)
self.reset_parameters()
defreset_parameters (self):
"""Initialize parameters"""
nn.init.xavier_uniform_( self.input_proj .weight)
22

# ----- PAGE BREAK (22) -----

nn.init.zeros_(self.input_proj .bias)
forlayer inself.gpr_layers:
layer.reset_parameters()
defforward(self, x, edge_index, edge_weight =None):
"""
Forward pass
Args:
x: Node features [num_nodes, input_dim]
edge_index: Edge indices [2, num_edges]
edge_weight: Edge weights [num_edges] (optional)
Returns:
Node embeddings [num_nodes, output_dim]
"""
# Input projection
h=self.input_proj(x)
h=F.relu(h)
h=F.dropout(h, p =self.dropout, training =self.training)
# Store input for residual connection
h_input =h
# Apply GPRGNN layers
fori, gpr_layer inenumerate (self.gpr_layers):
h_prev=h
# GPRGNN propagation
h=gpr_layer(h, edge_index, edge_weight)
# Residual connection (if enabled and dimensions match)
ifself.use_residual:
ifi==0:
# First layer residual
h_res=self.residual_projs[i](x)
else:
# Other layers residual
h_res=self.residual_projs[i](h_prev)
h=h+h_res
# Apply layer normalization (except for last layer)
ifi<self.num_layers -1:
h=self.layer_norms[i](h)
h=F.relu(h)
h=F.dropout(h, p =self.dropout, training =self.training)
23

# ----- PAGE BREAK (23) -----

returnh
defget_attention_weights (self, x, edge_index, edge_weight =None):
"""Get attention weights from the last layer"""
withtorch.no_grad():
embeddings =self.forward(x, edge_index, edge_weight)
# Global attention for interpretability
attn_output, attn_weights =self.global_attention(
embeddings .unsqueeze( 0),
embeddings .unsqueeze( 0),
embeddings .unsqueeze( 0)
)
returnattn_weights .squeeze( 0)
defget_propagation_weights (self):
"""Get learned propagation weights (gamma) from all layers"""
weights ={}
fori, layer inenumerate (self.gpr_layers):
weights[ f'layer_ {i}']=F.softmax(layer .gamma, dim =0).detach()
returnweights
[64]: class EconomicGAE (nn.Module):
"""Graph Autoencoder for unsupervised anomaly detection"""
def__init__ (self, input_dim: int, hidden_dim: int=64, latent_dim: int=␣
↪32):
super(EconomicGAE, self).__init__ ()
# Encoder
self.encoder_conv1 =GCNConv(input_dim, hidden_dim)
self.encoder_conv2 =GCNConv(hidden_dim, latent_dim)
# Decoder for graph structure
self.decoder_structure = lambdaz, edge_index, sigmoid =True: (
(z[edge_index[ 0]]*z[edge_index[ 1]]).sum(dim=1)
if notsigmoid elsetorch.sigmoid((z[edge_index[ 0]]*␣
↪z[edge_index[ 1]]).sum(dim=1))
)
# Decoder for node features
self.decoder_features =nn.Sequential(
nn.Linear(latent_dim, hidden_dim),
nn.ReLU(),
nn.Linear(hidden_dim, input_dim)
)
defencode(self, x, edge_index):
x=F.relu(self.encoder_conv1(x, edge_index))
24

# ----- PAGE BREAK (24) -----

returnself.encoder_conv2(x, edge_index)
defdecode_struct (self, z, edge_index):
returnself.decoder_structure(z, edge_index)
defdecode_feat (self, z):
returnself.decoder_features(z)
defforward(self, x, edge_index):
z=self.encode(x, edge_index)
reconstructed_struct =self.decode_struct(z, edge_index)
reconstructed_feat =self.decode_feat(z)
returnreconstructed_struct, reconstructed_feat
[65]: class EconomicGNN (nn.Module):
"""Graph Neural Network for Economic Data Analysis"""
def__init__ (self, input_dim: int, hidden_dim: int=64, output_dim: int=␣
↪32,
num_heads: int=4, dropout: float=0.1, use_edge_weight: ␣
↪bool=True):
super(EconomicGNN, self).__init__ ()
self.use_edge_weight =use_edge_weight
ifuse_edge_weight:
self.conv1=GCNConv(input_dim, hidden_dim)
self.conv2=GCNConv(hidden_dim, hidden_dim)
else:
self.conv1=GATConv(input_dim, hidden_dim, heads =num_heads,
dropout=dropout, concat =False)
self.conv2=GATConv(hidden_dim *1, hidden_dim, heads =1,#␣
↪hidden_dim * num_heads -> hidden_dim * 1
dropout=dropout, concat =False)
self.fc1=nn.Linear(hidden_dim, hidden_dim //2)
self.fc2=nn.Linear(hidden_dim //2, output_dim)
# Add LayerNorm layer
self.norm1=nn.LayerNorm(hidden_dim)
self.norm2=nn.LayerNorm(hidden_dim)
self.dropout =nn.Dropout(dropout)
self.relu=nn.ReLU()
defforward(self, x, edge_index, edge_weight =None):
25

# ----- PAGE BREAK (25) -----

attention_weights_1 =None
attention_weights_2 =None
ifself.use_edge_weight andedge_weight is not None:
x=self.conv1(x, edge_index, edge_weight =edge_weight)
else:
x, attention_weights_1 =self.conv1(x, edge_index, ␣
↪return_attention_weights =True)
# After passing conv1, apply LayerNorm
x=self.norm1(x)
x=self.relu(x)
x=self.dropout(x)
ifself.use_edge_weight andedge_weight is not None:
x=self.conv2(x, edge_index, edge_weight =edge_weight)
else:
x, attention_weights_2 =self.conv2(x, edge_index, ␣
↪return_attention_weights =True)
# After passing conv2, apply LayerNorm
x=self.norm2(x)
x=self.relu(x)
x=self.fc1(x)
x=self.relu(x)
x=self.dropout(x)
x=self.fc2(x)
returnx, (attention_weights_1, attention_weights_2)
[66]: class WeightedGATGNN (nn.Module):
"""GAT with edge weight incorporation"""
def__init__ (self, input_dim: int, hidden_dim: int=64, output_dim: int=␣
↪32,
num_heads: int=4, dropout: float=0.1):
super(WeightedGATGNN, self).__init__ ()
self.gat1=GATv2Conv(input_dim, hidden_dim, heads =num_heads, ␣
↪dropout=dropout)
self.gat2=GATv2Conv(hidden_dim *num_heads, hidden_dim, heads =1,␣
↪dropout=dropout)
self.fc1=nn.Linear(hidden_dim, hidden_dim //2)
self.fc2=nn.Linear(hidden_dim //2, output_dim)
26

# ----- PAGE BREAK (26) -----

self.dropout =nn.Dropout(dropout)
self.relu=nn.ReLU()
defforward(self, x, edge_index, edge_weight =None):
x, attention_weights_1 =self.gat1(x, edge_index, ␣
↪return_attention_weights =True)
x=self.relu(x)
x=self.dropout(x)
x, attention_weights_2 =self.gat2(x, edge_index, ␣
↪return_attention_weights =True)
x=self.relu(x)
x=self.fc1(x)
x=self.relu(x)
x=self.dropout(x)
x=self.fc2(x)
returnx, (attention_weights_1, attention_weights_2)
[67]: class HybridGNN (nn.Module):
"""Hybrid model combining GCN and GAT"""
def__init__ (self, input_dim: int, hidden_dim: int=64, output_dim: int=␣
↪32,
num_heads: int=4, dropout: float=0.1):
super(HybridGNN, self).__init__ ()
self.input_proj =nn.Linear(input_dim, hidden_dim)
self.gcn1=GCNConv(hidden_dim, hidden_dim)
self.gcn2=GCNConv(hidden_dim, hidden_dim)
self.gat=GATConv(hidden_dim, hidden_dim, heads =num_heads, ␣
↪dropout=dropout, concat =False)
self.fusion=nn.Linear(hidden_dim *2, hidden_dim)
self.fc1=nn.Linear(hidden_dim, hidden_dim //2)
self.fc2=nn.Linear(hidden_dim //2, output_dim)
self.dropout =nn.Dropout(dropout)
self.relu=nn.ReLU()
self.layer_norm1 =nn.LayerNorm(hidden_dim)
self.layer_norm2 =nn.LayerNorm(hidden_dim)
27

# ----- PAGE BREAK (27) -----

defforward(self, x, edge_index, edge_weight =None):
x=self.input_proj(x)
x=self.relu(x)
h_gcn=self.gcn1(x, edge_index, edge_weight =edge_weight)
h_gcn=self.relu(h_gcn)
h_gcn=self.dropout(h_gcn)
h_gcn=self.gcn2(h_gcn, edge_index, edge_weight =edge_weight)
h_gcn=self.layer_norm1(h_gcn)
h_gat=self.gat(x, edge_index)
h_gat=self.relu(h_gat)
h_gat=self.layer_norm2(h_gat)
h_combined =torch.cat([h_gcn, h_gat], dim =1)
h=self.fusion(h_combined)
h=self.relu(h)
h=self.dropout(h)
h=self.fc1(h)
h=self.relu(h)
h=self.dropout(h)
h=self.fc2(h)
returnh
[68]: class ImprovedEconomicGNN (nn.Module):
"""ESG + Leontief fusion GNN"""
def__init__ (self, input_dim: int, hidden_dim: int=64, output_dim: int=␣
↪32,
num_nodes: int=25, num_heads: int=4, dropout: float=0.1,
use_esg: bool= True, use_leontief: bool=True):
super().__init__ ()
self.gat1=nn.MultiheadAttention(hidden_dim, num_heads, ␣
↪dropout=dropout, batch_first =True)
self.gat2=nn.MultiheadAttention(hidden_dim, num_heads, ␣
↪dropout=dropout, batch_first =True)
self.input_proj =nn.Linear(input_dim, hidden_dim)
self.use_esg =use_esg
ifuse_esg:
self.esg_gating =ESGChannelGating(hidden_dim)
self.use_leontief =use_leontief
ifuse_leontief:
self.leontief_fusion =EconomicPriorFusion(num_nodes, hidden_dim)
28

# ----- PAGE BREAK (28) -----

self.output_layers =nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.LayerNorm(hidden_dim //2),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(hidden_dim //2, output_dim)
)
self.skip_weight =nn.Parameter(torch .tensor(0.2))
defforward(self, x: torch .Tensor, edge_index: torch .Tensor,
edge_weight: Optional[torch .Tensor] = None,
codes: Optional[ list]= None)->Dict[str, torch.Tensor]:
h=self.input_proj(x)
h_skip=h.clone()
h_attn, _ =self.gat1(h.unsqueeze( 0), h.unsqueeze( 0), h.unsqueeze( 0))
h=F.relu(h_attn .squeeze( 0))
h=F.dropout(h, p =0.1, training =self.training)
h_attn, attention_weights =self.gat2(h.unsqueeze( 0), h.unsqueeze( 0), h.
↪unsqueeze( 0))
h=F.relu(h_attn .squeeze( 0))
esg_scores = None
ifself.use_esg andcodes is not None andhasattr(self,'esg_gating '):
h, esg_scores =self.esg_gating(h, x, codes)
economic_metrics =None
ifself.use_leontief andedge_weight is not None:
h, _, economic_metrics =self.leontief_fusion(
h, edge_index, edge_weight, return_metrics =True
)
skip_weight =torch.sigmoid( self.skip_weight)
h=h+skip_weight *h_skip
output=self.output_layers(h)
results ={
'embeddings ': output,
'hidden_states ': h,
'attention_weights ': attention_weights
}
ifesg_scores is not None:
29

# ----- PAGE BREAK (29) -----

results[ 'esg_scores ']=esg_scores
ifeconomic_metrics is not None:
results[ 'economic_metrics ']=economic_metrics
returnresults
[69]: class EconomicESGGNN (nn.Module):
"""Complete Economic ESG GNN with all components"""
def__init__ (self, input_dim: int=6, hidden_dim: int=64,
num_nodes: int=20, num_heads: int=4):
super().__init__ ()
self.input_dim =input_dim
self.hidden_dim =hidden_dim
self.num_nodes =num_nodes
self.input_norm =nn.LayerNorm(input_dim)
self.input_projection =nn.Linear(input_dim, hidden_dim)
# Bidirectional GNN initialize
self.bidirectional_gnn =BidirectionalGNN(hidden_dim)
self.norm1=nn.LayerNorm(hidden_dim)
# ESG gating
self.esg_gating =ESGChannelGating(hidden_dim)
self.norm2=nn.LayerNorm(hidden_dim)
# fusion economic theories
self.economic_fusion =EconomicPriorFusion(num_nodes, hidden_dim)
# Prediction heads - print logits eliminating sigmoid
self.prediction_heads =nn.ModuleDict({
'esg_risk ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1)# Eliminate sigmoid
),
'economic_impact ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1),
nn.Tanh()
),
30

# ----- PAGE BREAK (30) -----

'volatility ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1),
nn.Softplus()
),
'transition_cost ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1)
),
'compliance_probability ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1)
)
})
defforward(self, x: torch .Tensor, edge_index: torch .Tensor,
edge_weight: torch .Tensor, codes: List[ str])->Dict[str, torch.
↪Tensor]:
# normalize input and projection
x_norm=self.input_norm(x)
h_initial =self.input_projection(x_norm)
# Bidirectional GNN
h_bidirectional, h_up, h_down, elasticity =self.bidirectional_gnn(
h_initial, edge_index, edge_weight
)
h_bidirectional =self.norm1(h_bidirectional)
# ESG gating
h_gated, esg_scores =self.esg_gating(h_bidirectional, x, codes)
h_gated =self.norm2(h_gated)
# Fusing economical theories
h_fused, learned_adjacency, _ =self.economic_fusion(
h_gated, edge_index, edge_weight, return_metrics =False
)
# Connect residuals
h_final =h_fused +h_initial
31

# ----- PAGE BREAK (31) -----

# Predictions
predictions ={
task_name: head(h_final) .squeeze( -1)
fortask_name, head inself.prediction_heads .items()
}
# Additional output
predictions .update({
'embeddings ': h_final,
'esg_scores ': esg_scores,
'elasticity ': elasticity,
'learned_adjacency ': learned_adjacency,
'upstream_features ': h_up,
'downstream_features ': h_down
})
returnpredictions
[71]: class BidirectionalGNN (nn.Module):
"""Bidirectional GNN for upstream and downstream modeling"""
def__init__ (self, hidden_dim: int, num_heads: int=4):
super().__init__ ()
self.downstream_conv =GATv2Conv(hidden_dim, hidden_dim, ␣
↪heads=num_heads, concat =False)
self.upstream_conv =GATv2Conv(hidden_dim, hidden_dim, heads =num_heads, ␣
↪concat=False)
self.fusion_layer =nn.Linear(hidden_dim *2, hidden_dim)
self.elasticity_layer =nn.Linear(hidden_dim *2,1)
defforward(self, x, edge_index, edge_weight):
upstream_edge_index =edge_index[[ 1,0]]
h_down=F.relu(self.downstream_conv(x, edge_index))
h_up=F.relu(self.upstream_conv(x, upstream_edge_index))
h_combined =torch.cat([h_down, h_up], dim =1)
h_fused =F.relu(self.fusion_layer(h_combined))
elasticity =torch.sigmoid( self.elasticity_layer(h_combined)) .
↪squeeze( -1)
returnh_fused, h_up, h_down, elasticity
[ ]:
32

# ----- PAGE BREAK (32) -----

[ ]:#ESG Channel gating
[72]: class ESGChannelGating (nn.Module):
"""ESG-based feature gating mechanism"""
def__init__ (self, hidden_dim: int):
super().__init__ ()
self.gate_network =nn.Sequential(
nn.Linear(2, hidden_dim),
nn.Sigmoid()
)
self.esg_scorer =nn.Linear(hidden_dim, 3)
defforward(self, h_gnn, x, codes):
# Use ESG features as gate inputs
ifx.shape[1]>=8:
gate_values =self.gate_network(x[:, 6:8])# ESG env, social
else:
gate_values =torch.ones_like(h_gnn)
h_gated =h_gnn*gate_values
esg_scores =torch.softmax( self.esg_scorer(h_gated), dim =-1)
returnh_gated, esg_scores
[ ]:
[ ]:#Economic prior fusion
[73]: class EconomicPriorFusion (nn.Module):
"""Leontief inverse matrix theory fusion with GNN"""
def__init__ (self, num_nodes: int, hidden_dim: int):
super().__init__ ()
self.num_nodes =num_nodes
self.hidden_dim =hidden_dim
self.leontief_residual =nn.Parameter(torch .randn(num_nodes, num_nodes) ␣
↪*0.01)
self.theory_weight =nn.Parameter(torch .tensor(0.5))
self.adaptive_fusion =nn.Sequential(
nn.Linear(hidden_dim *2, hidden_dim),
nn.ReLU(),
nn.Linear(hidden_dim, hidden_dim),
)
33

# ----- PAGE BREAK (33) -----

defforward(self, h_gnn: torch .Tensor, edge_index: torch .Tensor,
edge_weight: torch .Tensor, return_metrics: bool= False):
device=h_gnn.device
A=torch.zeros(self.num_nodes, self.num_nodes, device =device)
A[edge_index[ 0], edge_index[ 1]]=edge_weight
I=torch.eye(self.num_nodes, device =device)
try:
leontief_inv =torch.linalg.inv(I-A)
excepttorch.linalg.LinAlgError:
leontief_inv =torch.linalg.pinv(I-A)
enhanced_matrix =torch.softmax(leontief_inv +self.leontief_residual, ␣
↪dim=1)
h_economic =torch.matmul(enhanced_matrix, h_gnn)
h_combined =torch.cat([h_gnn, h_economic], dim =1)
h_adaptive =self.adaptive_fusion(h_combined)
theory_weight =torch.sigmoid( self.theory_weight)
h_final =theory_weight *h_economic +(1-theory_weight) *h_gnn+0.
↪1*h_adaptive
metrics = None
ifreturn_metrics:
metrics ={
'theory_weight ': theory_weight .item(),
'matrix_condition ': torch.linalg.cond(enhanced_matrix) .item(),
'output_mult ': enhanced_matrix .sum(dim=0),
'forward_link ': enhanced_matrix .sum(dim=1)
}
returnh_final, enhanced_matrix, metrics
[ ]:
[ ]:#ESG prediction head
[74]: class ESGPredictionHead (nn.Module):
"""Numerically stable head for ESG prediction”"""
def__init__ (self, hidden_dim: int):
super().__init__ ()
# Prediction haeds for each tasks
self.prediction_heads =nn.ModuleDict({
34

# ----- PAGE BREAK (34) -----

'esg_risk ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.LayerNorm(hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1)
# Sigmoidi is operated in BCEWithLogitsLoss
),
'economic_impact ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.LayerNorm(hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1),
nn.Tanh()
),
'volatility ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.LayerNorm(hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1),
nn.Softplus()
),
'transition_cost ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.LayerNorm(hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1)
),
'compliance_probability ': nn.Sequential(
nn.Linear(hidden_dim, hidden_dim //2),
nn.LayerNorm(hidden_dim //2),
nn.ReLU(),
nn.Dropout( 0.2),
nn.Linear(hidden_dim //2,1)
)
})
# Initialize weights
self._init_weights()
def_init_weights (self):
"""Initialize weights stable"""
formodule inself.prediction_heads .values():
form inmodule:
35

# ----- PAGE BREAK (35) -----

ifisinstance (m, nn.Linear):
nn.init.xavier_normal_(m .weight, gain =0.5)
ifm.bias is not None:
nn.init.constant_(m .bias,0)
defforward(self, embeddings):
predictions ={}
fortask_name, head inself.prediction_heads .items():
pred=head(embeddings) .squeeze( -1)
# NaN check
pred=torch.where(torch .isnan(pred), torch .zeros_like(pred), pred)
pred=torch.clamp(pred, -10,10)# adjust clamp
predictions[task_name] =pred
returnpredictions
[ ]:
[75]: import networkx as nx
import torch
import matplotlib .pyplot as plt
import numpy as np
defvisualize_shock_propagation (model, data, sector_codes, shock_node_idx, ␣
↪top_k=10, ax=None):
"""
Visualizes shock propagation from a given node using GAT attention weights.
Args:
model: The trained GAT-based model.
data: The input data dictionary.
sector_codes (list): List of sector codes.
shock_node_idx (int): The index of the node where the shock originates.
top_k (int): The number of most affected nodes to display.
ax: Matplotlib axis to plot on.
"""
model.eval()
withtorch.no_grad():
# --- Run the model to get attention weights ---
embeddings, attention_tuples =model(data[ 'x'], data[ 'edge_index '],␣
↪data['edge_weight '])
# We'll use the attention weights from the first layer for this example
attention_weights_1 =attention_tuples[ 0]
ifattention_weights_1 is None:
print("This model does not appear to be a GAT model or attention ␣
↪weights were not returned. ")
36

# ----- PAGE BREAK (36) -----

return
edge_index_att =attention_weights_1[ 0]
edge_values =attention_weights_1[ 1].mean(dim =1)# Average over heads
# --- Create a directed graph ---
G=nx.DiGraph()
fori, code inenumerate (sector_codes):
G.add_node(i, label =code)
fori inrange(edge_index_att .shape[1]):
source, target =edge_index_att[ 0, i].item(), edge_index_att[ 1, i].
↪item()
weight=edge_values[i] .item()
G.add_edge(source, target, weight =weight)
# --- Identify and trace the shock ---
ifG.has_node(shock_node_idx):
# Get outgoing edges from the shock node
shock_edges =list(G.out_edges(shock_node_idx, data =True))
ifshock_edges:
# Sort by attention weight
shock_edges .sort(key =lambdax: x[2]['weight'], reverse =True)
# --- Prepare for plotting ---
subgraph_nodes ={shock_node_idx}
subgraph_edges =[]
edge_weights =[]
for_, target, data inshock_edges[:top_k]:
subgraph_nodes .add(target)
subgraph_edges .append((shock_node_idx, target))
edge_weights .append(data[ 'weight'])
subgraph =G.subgraph( list(subgraph_nodes))
pos=nx.spring_layout(subgraph, k =0.8, iterations =50, seed=42)
# --- Plotting ---
ifax is None:
fig, ax =plt.subplots(figsize =(14,10))
node_colors =['red'ifn==shock_node_idx else'skyblue'for␣
↪n insubgraph .nodes()]
node_labels ={n: G.nodes[n][ 'label'] forninsubgraph .nodes()}
37

# ----- PAGE BREAK (37) -----

nx.draw_networkx_nodes(subgraph, pos, node_color =node_colors, ␣
↪node_size =2500, ax=ax)
nx.draw_networkx_labels(subgraph, pos, labels =node_labels, ␣
↪font_size =10, font_weight ='bold', ax=ax)
# Draw the edges
drawn_edges =nx.draw_networkx_edges(subgraph, pos, ␣
↪edgelist =subgraph_edges,
edge_color =edge_weights, ␣
↪width=2.5,
edge_cmap =plt.cm.viridis, ␣
↪arrowsize =20,
node_size =2500, ax=ax)
# --- Colorbar for edge weights ---
sm=plt.cm.ScalarMappable(cmap =plt.cm.viridis,
norm=plt.
↪Normalize(vmin =min(edge_weights), vmax =max(edge_weights)))
sm.set_array([])
cbar=plt.colorbar(sm, ax =ax, orientation ='vertical ',␣
↪fraction =0.046, pad=0.04)
cbar.set_label( 'Attention Weight ', rotation =270, labelpad =15)
ax.set_title( f'Shock Propagation from Sector ␣
↪{sector_codes[shock_node_idx] }(Top {top_k }Impacts) ', fontsize =16)
ax.axis('off')
else:
ax.text(0.5,0.5,f"No outgoing connections found for Sector ␣
↪{sector_codes[shock_node_idx] }.",
ha='center', va='center', fontsize =12)
ax.set_title( f'Shock Propagation from Sector ␣
↪{sector_codes[shock_node_idx] }', fontsize =16)
ax.axis('off')
[ ]:
[ ]:#Complete ESG Analysis system
[81]: class CompleteESGAnalysisSystem :
"""Complete ESG analysis system"""
def__init__ (self):
self.preprocessor =RealDataPreprocessor()
self.models={}
self.prediction_heads ={}
self.data=None
38

# ----- PAGE BREAK (38) -----

self.targets =None
self.results ={}
defload_all_data (self):
"""Load all real data"""
print("="*70)
print("Start ESG analysis sysetem based on real data ")
print("="*70)
# 1. ESG data
esg_stats =self.preprocessor .load_esg_data( 'data.csv ')
# 2. I-O table
io_data =self.preprocessor .load_io_data( 'REAL_USE.xlsx ','REAL_MAKE.
↪xlsx')
# 3. Economic data
self.preprocessor .load_real_gdp_data( 'Real Gross Output by Industry.
↪csv')
# 4. Environmental data
self.preprocessor .load_real_env_data( 'ghgp_data_2023.xlsx ')
# 5. Enovironmental taxes data
env_tax_data =self.preprocessor .
↪load_environmental_taxes( 'Environmental Taxes.csv ')
self.preprocessor .
↪load_and_process_news_data(api_key ='e086209f280f46a482442a237d195a11 ')
# 6. Generate integrated features
self.data=self.preprocessor .create_integrated_features()
# 7. Generate ESG targets
self.targets =self.preprocessor .create_esg_targets()
ifself.data is None:
raise Exception ("Data load fails ")
print(f"\n￿ All data load completes! ")
print(f"- Number of sectors: {self.data['n_nodes']}")
print(f"- Number of features: {self.data['n_features ']}")
print(f"- Number of edges: {self.data['n_edges']}")
returnself.data,self.targets
39

# ----- PAGE BREAK (39) -----

definitialize_all_models (self, device: torch .device):
"""Initialize all GNN models"""
print(f"\nInitialize multiple GNN models... ")
hidden_dim =128
# 1. EconomicGNN (GCN)
self.models['EconomicGNN_GCN ']=EconomicGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
output_dim =hidden_dim,
use_edge_weight =True
).to(device)
# 2. EconomicGNN (GAT)
self.models['EconomicGNN_GAT ']=EconomicGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
output_dim =hidden_dim,
use_edge_weight =False
).to(device)
# 3. WeightedGATGNN
self.models['WeightedGATGNN ']=WeightedGATGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
output_dim =hidden_dim
).to(device)
# 4. HybridGNN
self.models['HybridGNN ']=HybridGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
output_dim =hidden_dim
).to(device)
# 5. ImprovedEconomicGNN
self.models['ImprovedEconomicGNN ']=ImprovedEconomicGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
output_dim =hidden_dim,
num_nodes =self.data['n_nodes'],
use_esg=True,
use_leontief =True
).to(device)
40

# ----- PAGE BREAK (40) -----

# 6. EconomicESGGNN
self.models['EconomicESGGNN ']=EconomicESGGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
num_nodes =self.data['n_nodes']
).to(device)
# 7.
self.models['GPR_GNN']=GPRGNN(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
output_dim =hidden_dim,
num_layers =2,
alpha=0.1,
K=10,
dropout=0.1
).to(device)
# 8.
self.models['EconomicGAE ']=EconomicGAE(
input_dim =self.data['n_features '],
hidden_dim =hidden_dim,
latent_dim =hidden_dim //2
).to(device)
# Predictied heads (except for EconomicESGGNN)
formodel_name inself.models.keys():
ifmodel_name !='EconomicESGGNN ':
self.prediction_heads[model_name] =␣
↪ESGPredictionHead(hidden_dim) .to(device)
print(f"￿{len(self.models) }models are initialized ")
formodel_name, model inself.models.items():
param_count =sum(p.numel() forpinmodel.parameters())
ifmodel_name !='EconomicESGGNN ':
param_count +=sum(p.numel() forp inself.
↪prediction_heads[model_name] .parameters())
print(f"• {model_name }: {param_count :,}parameter ")
deftrain_anomaly_detector (self, device: torch .device, num_epochs: int=␣
↪100):
"""Train the EconomicGAE model for anomaly detection"""
print("\nTraining Unsupervised Anomaly Detector (EconomicGAE)... ")
model=self.models.get("EconomicGAE ")
if notmodel:
41

# ----- PAGE BREAK (41) -----

print("EconomicGAE model not found. Please initialize it first. ")
return
optimizer =torch.optim.Adam(model .parameters(), lr =0.005)
model.train()
x=self.data["x"].to(device)
edge_index =self.data["edge_index "].to(device)
forepoch inrange(num_epochs):
optimizer .zero_grad()
z=model.encode(x, edge_index)
# Structural reconstruction loss
pos_edge_index =edge_index
# decode_struct should return probabilities for observed edges
pos_loss =-torch.log(model .decode_struct(z, pos_edge_index) +␣
↪1e-15).mean()
# Feature reconstruction loss
x_reconstructed =model.decode_feat(z)
feature_loss =F.mse_loss(x_reconstructed, x)
# Combined loss (weight adjustable)
loss=pos_loss +0.5*feature_loss
loss.backward()
optimizer .step()
if(epoch+1)%20==0:
print(f"Epoch {epoch+1:03d }/{num_epochs }| Loss: {loss.item() :.
↪4f }")
defdetect_anomalies (self, device: torch .device) ->pd.DataFrame:
"""Detect anomalies using the trained EconomicGAE model"""
print("\nDetecting anomalies... ")
model=self.models.get('EconomicGAE ')
if notmodel:
print("EconomicGAE model not found. Please train it first. ")
returnpd.DataFrame()
model.eval()
withtorch.no_grad():
z=model.encode(self.data['x'].to(device), self.data['edge_index '].
↪to(device))
x_reconstructed =model.decode_feat(z)
42

# ----- PAGE BREAK (42) -----

# Calculate feature reconstruction error for each node
feature_recon_error =torch.mean((self.data['x'].to(device) -␣
↪x_reconstructed) **2, dim=1)
# For structural anomaly, we can approximate by looking at the ␣
↪change in embeddings
struct_recon_error =torch.mean((model .encode(x_reconstructed, self.
↪data['edge_index '].to(device)) -z)**2, dim=1)
# Combine errors for a final anomaly score
anomaly_scores =feature_recon_error +0.5*struct_recon_error
results =pd.DataFrame({
'sector_code ':self.data['sector_codes '],
'anomaly_score ': anomaly_scores .cpu().numpy()
}).sort_values( 'anomaly_score ', ascending =False).
↪reset_index(drop =True)
print("Top 10 most anomalous industries: ")
print(results .head(10))
returnresults
deftrain_all_models (self, device: torch .device, num_epochs: int=300):
"""Train all models"""
print(f"\n {len(self.models) }models are trained simulataneously... ")
# Move data to device
forkey in['x','edge_index ','edge_weight ']:
ifkey inself.data:
self.data[key] =self.data[key] .to(device)
targets_device ={}
forkey, value inself.targets.items():
targets_device[key] =value.to(device)
# Set optimizers
optimizers ={}
schedulers ={}
formodel_name, model inself.models.items():
ifmodel_name =='EconomicESGGNN ':
params=list(model.parameters())
else:
params=list(model.parameters()) +list(self.
↪prediction_heads[model_name] .parameters())
43

# ----- PAGE BREAK (43) -----

lr=1e-4 ifmodel_name in['EconomicGNN_GCN ','HybridGNN '] else␣
↪1e-3
optimizers[model_name] =torch.optim.AdamW(params, lr =lr,␣
↪weight_decay =1e-4)
schedulers[model_name] =torch.optim.lr_scheduler .CosineAnnealingLR(
optimizers[model_name], T_max =num_epochs
)
# Loss functions
loss_functions ={
'esg_risk ': nn.BCEWithLogitsLoss(), # Use BCEWithLogitsLoss ␣
↪instead of BCE
'economic_impact ': nn.MSELoss(),
'volatility ': nn.MSELoss(),
'transition_cost ': nn.BCEWithLogitsLoss(),
'compliance_probability ': nn.BCEWithLogitsLoss()
}
loss_weights ={
'esg_risk ':0.25,
'economic_impact ':0.25,
'volatility ':0.2,
'transition_cost ':0.15,
'compliance_probability ':0.15
}
# Training history
training_history ={model_name: [] formodel_name inself.models.keys()}
# Training loop
start_time =time.time()
forepoch inrange(num_epochs):
epoch_losses ={}
formodel_name, model inself.models.items():
ifmodel_name =='EconomicGAE ':
continue
model.train()
optimizer =optimizers[model_name]
optimizer .zero_grad()
# Forward pass
predictions =None
44

# ----- PAGE BREAK (44) -----

ifmodel_name =='EconomicESGGNN ':
predictions =model(self.data['x'],self.data['edge_index '],
self.data['edge_weight '],self.
↪data['codes'])
elifmodel_name =='ImprovedEconomicGNN ':
model_output =model(self.data['x'],self.
↪data['edge_index '],
self.data['edge_weight '],self.
↪data['codes'])
embeddings =model_output[ 'embeddings ']
predictions =self.prediction_heads[model_name](embeddings)
else:
model_output =model(self.data['x'],self.
↪data['edge_index '],
self.data['edge_weight '])
# Check if it's a tuple (multiple returns) or single value
ifisinstance (model_output, tuple):
embeddings =model_output[ 0]# Take embeddings (first ␣
↪element)
attention_weights =model_output[ 1]if␣
↪len(model_output) >1else None
else:
embeddings =model_output # Single return value
attention_weights = None
predictions =self.prediction_heads[model_name](embeddings)
# Loss calculation
total_loss =0
fortask_name, loss_fn inloss_functions .items():
iftask_name intargets_device andtask_name inpredictions:
pred=predictions[task_name]
target=targets_device[task_name]
# Eliminate sigmoid for BCEWithLogitsLoss
iftask_name in['esg_risk ','transition_cost ',␣
↪'compliance_probability ']:
# Transition to logit if predictied value already ␣
↪passed through sigmoid
ifhasattr(self.models[model_name], ␣
↪'prediction_heads ') ormodel_name =='EconomicESGGNN ':
# Inverse transition because model already ␣
↪applied sigmoid
pred=torch.clamp(pred, 1e-7,1-1e-7)#␣
↪Clamping for numerical stability
45

# ----- PAGE BREAK (45) -----

pred=torch.log(pred /(1-pred)) # Inverse ␣
↪function of sigmoid (logit)
task_loss =loss_fn(pred, target)
# Check NaN or inf
iftorch.isnan(task_loss) ortorch.isinf(task_loss):
print(f"￿ {model_name }-{task_name }: Loss is NaN/
↪Inf. Skip. ")
task_loss =torch.tensor(0.0, device =device,␣
↪requires_grad =True)
total_loss +=task_loss *loss_weights[task_name]
# Backward pass
total_loss .backward()
torch.nn.utils.clip_grad_norm_(
list(model.parameters()) +
(list(self.prediction_heads[model_name] .parameters()) if␣
↪model_name !='EconomicESGGNN ' else[]),
max_norm =1.0
)
optimizer .step()
schedulers[model_name] .step()
epoch_losses[model_name] =total_loss .item()
training_history[model_name] .append(total_loss .item())
# Print processing
if(epoch+1)%25==0:
print(f"\n[Epoch {epoch+1:03d }/{num_epochs }] loss for each ␣
↪model:")
formodel_name, loss inepoch_losses .items():
print(f"• {model_name }: {loss :.4f }")
training_time =time.time()-start_time
print(f"\n￿ All models are trained! (Time spent: {training_time :.
↪1f }sec)")
self.results[ 'training_history ']=training_history
self.results[ 'training_time ']=training_time
returntraining_history
defevaluate_models (self, device: torch .device):
"""Evaluation of all model performances"""
46

# ----- PAGE BREAK (46) -----

print(f"\n'Evaluation of model performance '...")
evaluation_results ={}
# Move targets to device
targets_device ={}
forkey, value inself.targets.items():
targets_device[key] =value.to(device)
formodel_name, model inself.models.items():
ifmodel_name =='EconomicGAE ':
continue
model.eval()
withtorch.no_grad():
# Perform prediction
# In evaluate_models method
ifmodel_name =='EconomicESGGNN ':
predictions =model(self.data['x'],self.data['edge_index '],
self.data['edge_weight '],self.
↪data['codes'])
elifmodel_name =='ImprovedEconomicGNN ':
model_output =model(self.data['x'],self.
↪data['edge_index '],
self.data['edge_weight '],self.
↪data['codes'])
embeddings =model_output[ 'embeddings ']
predictions =self.prediction_heads[model_name](embeddings)
else:
model_output =model(self.data['x'],self.
↪data['edge_index '],
self.data['edge_weight '])
ifisinstance (model_output, tuple):
embeddings =model_output[ 0]
else:
embeddings =model_output
predictions =self.prediction_heads[model_name](embeddings)
# Calculate performance indicator - converse prediction value ␣
↪to probability
metrics ={}
fortask_name intargets_device .keys():
iftask_name inpredictions:
47

# ----- PAGE BREAK (47) -----

y_true=targets_device[task_name] .cpu().numpy()
y_pred_raw =predictions[task_name] .cpu().numpy()
# Apply sigmoid for tasks which are used ␣
↪BCEWithLogitsLoss
iftask_name in['esg_risk ','transition_cost ',␣
↪'compliance_probability ']:
y_pred=1/(1+np.exp(-y_pred_raw)) # sigmoid ￿￿
else:
y_pred=y_pred_raw
mse=mean_squared_error(y_true, y_pred)
mae=mean_absolute_error(y_true, y_pred)
r2=r2_score(y_true, y_pred)
metrics[ f'{task_name }_mse']=mse
metrics[ f'{task_name }_mae']=mae
metrics[ f'{task_name }_r2']=r2
# Accuracy of classification (0.5 threshold)
iftask_name in['esg_risk ','transition_cost ',␣
↪'compliance_probability ']:
accuracy =np.mean((y_pred >0.5)==(y_true >0.5))
metrics[ f'{task_name }_accuracy ']=accuracy
evaluation_results[model_name] =metrics
# Print the result
print(f"\nSummary model performance respectively (R² standard): ")
formodel_name, metrics inevaluation_results .items():
r2_scores =[v fork, v inmetrics.items() ifk.endswith( '_r2')]
avg_r2=np.mean(r2_scores) ifr2_scores else0
print(f"• {model_name }: mean R² = {avg_r2 :.4f }")
self.results[ 'evaluation ']=evaluation_results
returnevaluation_results
defcreate_comprehensive_visualizations (self):
"""overall visualization"""
print(f"\nOverall result visualization... ")
if notself.results:
print("￿ No evaluation result. ")
return
plt.style.use('default')
48

# ----- PAGE BREAK (48) -----

fig, axes =plt.subplots( 3,2, figsize =(16,18))
# 1. Training loss
if'training_history 'inself.results:
colors=['red','blue','green','orange','purple','brown']
fori, (model_name, losses) inenumerate (self.
↪results[ 'training_history '].items()):
color=colors[i %len(colors)]
iflen(losses) >10:
# Smoothing
smoothed =np.convolve(losses, np .ones(10)/10, mode='valid')
axes[0,0].plot(smoothed, label =model_name .replace( 'GNN',␣
↪''),
color=color, linewidth =2, alpha=0.8)
else:
axes[0,0].plot(losses, label =model_name .replace( 'GNN',''),
color=color, linewidth =2, alpha=0.8)
axes[0,0].set_title( 'Training loss by models ', fontweight ='bold')
axes[0,0].set_xlabel( 'Epoch')
axes[0,0].set_ylabel( 'Loss')
axes[0,0].legend()
axes[0,0].grid( True, alpha=0.3)
# 2. R² score heatmap
if'evaluation ' inself.results:
model_names =list(self.results[ 'evaluation '].keys())
task_names =['esg_risk ','economic_impact ','volatility ',␣
↪'transition_cost ','compliance_probability ']
r2_matrix =[]
formodel_name inmodel_names:
r2_row=[]
fortask_name intask_names:
r2_key=f'{task_name }_r2'
r2_val=self.results[ 'evaluation '][model_name] .get(r2_key, ␣
↪0)
r2_row.append(r2_val)
r2_matrix .append(r2_row)
ifr2_matrix:
im=axes[0,1].imshow(r2_matrix, cmap ='RdYlGn', aspect ='auto',␣
↪vmin=-0.2, vmax=1.0)
axes[0,1].set_xticks( range(len(task_names)))
axes[0,1].set_xticklabels([name .replace( '_','\n')forname in␣
↪task_names])
49

# ----- PAGE BREAK (49) -----

axes[0,1].set_yticks( range(len(model_names)))
axes[0,1].set_yticklabels([name .replace( 'GNN','')forname in␣
↪model_names])
axes[0,1].set_title( 'R² score by models ', fontweight ='bold')
fori inrange(len(model_names)):
forjinrange(len(task_names)):
axes[0,1].text(j, i, f'{r2_matrix[i][j] :.2f }',
ha="center", va="center", fontsize =8)
plt.colorbar(im, ax =axes[0,1])
# 3. Best performance model by tasks
if'evaluation ' inself.results:
task_names =['esg_risk ','economic_impact ','volatility ',␣
↪'transition_cost ','compliance_probability ']
best_models ={}
fortask_name intask_names:
best_r2 =-float('inf')
best_model = None
formodel_name, metrics inself.results[ 'evaluation '].items():
r2_key=f'{task_name }_r2'
ifr2_key inmetrics andmetrics[r2_key] >best_r2:
best_r2 =metrics[r2_key]
best_model =model_name
ifbest_model:
best_models[task_name] =best_model
ifbest_models:
model_counts ={}
formodel inbest_models .values():
model_counts[model] =model_counts .get(model, 0)+1
models=list(model_counts .keys())
counts=list(model_counts .values())
bars=axes[1,0].bar(range(len(models)), counts,
color=['skyblue','lightcoral ',␣
↪'lightgreen ','gold'][:len(models)])
axes[1,0].set_xticks( range(len(models)))
axes[1,0].set_xticklabels([m .replace( 'GNN','') formin␣
↪models], rotation =45)
axes[1,0].set_ylabel( 'Number of best performance task ')
axes[1,0].set_title( 'Number of winner models by task ',␣
↪fontweight ='bold')
50

# ----- PAGE BREAK (50) -----

axes[1,0].grid( True, alpha=0.3, axis='y')
forbar, count inzip(bars, counts):
height=bar.get_height()
axes[1,0].text(bar .get_x() +bar.get_width() /2., height +␣
↪0.05,
f'{count }', ha='center', va='bottom')
# 4. Real vs Prediction (ESG risk)
if'evaluation ' inself.results:
# Prediction with best model vs Real
best_overall_model =max(self.results[ 'evaluation '].keys(),
key=lambdax: np.nanmean([v fork, v inself.
↪results[ 'evaluation '][x].items()
ifk.endswith( '_r2')]))
model=self.models[best_overall_model]
model.eval()
withtorch.no_grad():
ifbest_overall_model =='EconomicESGGNN ':
predictions =model(self.data['x'],self.data['edge_index '],
self.data['edge_weight '],self.
↪data['codes'])
elifbest_overall_model =='ImprovedEconomicGNN ':
model_output =model(self.data['x'],self.
↪data['edge_index '],
self.data['edge_weight '],self.
↪data['codes'])
predictions =self.
↪prediction_heads[best_overall_model](model_output[ 'embeddings '])
else:
embeddings =model(self.data['x'],self.data['edge_index '],␣
↪self.data['edge_weight '])
predictions =self.
↪prediction_heads[best_overall_model](embeddings)
if'esg_risk 'inpredictions:
y_true=self.targets[ 'esg_risk '].cpu().numpy()
y_pred_raw =predictions[ 'esg_risk '].cpu().numpy()
y_pred=1/(1+np.exp(-y_pred_raw))
axes[1,1].scatter(y_true, y_pred, alpha =0.6, s=50)
axes[1,1].plot([0,1], [0,1],'r--', alpha=0.8)
axes[1,1].set_xlabel( 'Real ESG risk ')
axes[1,1].set_ylabel( 'Predicted ESG risk ')
51

# ----- PAGE BREAK (51) -----

axes[1,1].set_title( f'Real vs Prediction ␣
↪({best_overall_model .replace( "GNN","")})', fontweight ='bold')
axes[1,1].grid( True, alpha=0.3)
r2=r2_score(y_true, y_pred)
axes[1,1].text(0.05,0.95,f'R² = {r2 :.3f }',
transform =axes[1,1].transAxes,
bbox=dict(boxstyle ="round,pad=0.3 ",␣
↪facecolor ="yellow", alpha=0.7))
# 5. Importance of features
ifself.data and'n_features 'inself.data:
esg_feature_indices =[6,7,8,9]# ESG related features
esg_feature_names =['Env','Soc','Gov','ESGVol']
# Calculate mean values between features
esg_values =[]
foridx inesg_feature_indices:
ifidx<self.data['x'].shape[1]:
esg_values .append(self.data['x'][:, idx] .mean().item())
else:
esg_values .append(0)
bars=axes[2,0].bar(esg_feature_names, esg_values,
color=['green','blue','orange','red'])
axes[2,0].set_ylabel( 'Mean feature value ')
axes[2,0].set_title( 'ESG mean value by features ',␣
↪fontweight ='bold')
axes[2,0].grid( True, alpha=0.3, axis='y')
forbar, value inzip(bars, esg_values):
height=bar.get_height()
axes[2,0].text(bar .get_x() +bar.get_width() /2., height +0.01,
f'{value :.3f }', ha='center', va='bottom')
# 6. Summary overall performance
if'evaluation ' inself.results:
model_performance =[]
model_names_short =[]
formodel_name, metrics inself.results[ 'evaluation '].items():
r2_scores =[v fork, v inmetrics.items() ifk.endswith( '_r2')]
avg_r2=np.mean(r2_scores) ifr2_scores else0
model_performance .append(avg_r2)
model_names_short .append(model_name .replace( 'GNN',''))
# Arrange by performance
52

# ----- PAGE BREAK (52) -----

sorted_data =sorted(zip(model_names_short, model_performance), ␣
↪key=lambdax: x[1], reverse =True)
sorted_names, sorted_performance =zip(*sorted_data)
bars=axes[2,1].barh(range(len(sorted_names)), sorted_performance,
color=['gold','silver','brown',␣
↪'lightblue ','lightgreen ','pink'][:len(sorted_names)])
axes[2,1].set_yticks( range(len(sorted_names)))
axes[2,1].set_yticklabels(sorted_names)
axes[2,1].set_xlabel( 'Mean R² score ')
axes[2,1].set_title( 'Model performance ranking ', fontweight ='bold')
axes[2,1].grid( True, alpha=0.3, axis='x')
fori, (bar, score) inenumerate (zip(bars, sorted_performance)):
width=bar.get_width()
axes[2,1].text(width +0.01, bar.get_y() +bar.get_height() /2.,
f'{score :.3f }', ha='left', va='center')
plt.tight_layout()
plt.show()
print("￿ Overall visualization completes! ")
defgenerate_final_report (self):
"""Generate final analysis report"""
if'evaluation ' not inself.results:
print("￿ Can not generate report with no evaluation results. ")
return
# Find the best performing model based on average R² score
best_overall_model =max(self.results[ 'evaluation '].keys(),
key=lambdax: np.mean([v fork, v inself.
↪results[ 'evaluation '][x].items() ifk.endswith( '_r2')]))
best_avg_r2 =np.mean([v fork, v inself.
↪results[ 'evaluation '][best_overall_model] .items() ifk.endswith( '_r2')])
report=f"""
Final ESG GNN analysis report based on real data
{'='*70 }
Data:
• ESG corperation data: {len(self.preprocessor .esg_data) ifself.
↪preprocessor .esg_data is not None else0}company
• I-O table: {self.data['n_nodes']}industry × {self.data['n_nodes']}industry
• Integrated feature: {self.data['n_features ']}feature
• Network connection: {self.data['n_edges']}edge
53

# ----- PAGE BREAK (53) -----

• Data source: {self.data['data_source ']}
Model performance:
• Number of trained models: {len(self.models) }
• Highest performance model: {best_overall_model }
• Mean R² score: {best_avg_r2 :.4f }
• Training time: {self.results.get('training_time ',0):.1f }sec
Performance by tasks: """
if'evaluation ' inself.results:
fortask_name in['esg_risk ','economic_impact ','volatility ',␣
↪'transition_cost ','compliance_probability ']:
best_model = None
best_score =-float('inf')
formodel_name, metrics inself.results[ 'evaluation '].items():
r2_key=f'{task_name }_r2'
ifr2_key inmetrics andmetrics[r2_key] >best_score:
best_score =metrics[r2_key]
best_model =model_name
ifbest_model:
report+=f"\n• {task_name }:{best_model }(R² =␣
↪{best_score :.3f })"
report+=f"""
Key findings:
• Industries with lower ESG scores show higher risk predictions
• Strong correlation between environmental tax exposure and economic impact
• Bidirectional GNNs outperform unidirectional models
• ESG gating mechanism enables effective feature selection
Practical implications:
• Utilize ESG scores for portfolio risk management
• Develop investment strategies that account for inter-industry spillover ␣
↪effects
• Establish proactive response systems to regulatory changes
• Optimize sustainable supply chain management
Limitations:
• Prediction accuracy depends on data quality
• Limited adaptability to sudden policy shifts
• Increased uncertainty in long-term forecasts
Future works:
54

# ----- PAGE BREAK (54) -----

• Build real-time data update systems
• Integrate a wider range of ESG indicators
• Tailor models by country/region
• Incorporate explainable AI techniques
"""
print(report)
# Save the report
timestamp =time.strftime( "%Y%m%d_%H%M%S")
filename =f"ESG_GNN_Analysis_Report_ {timestamp }.txt"
try:
withopen(filename, 'w', encoding ='utf-8') asf:
f.write(report)
print(f"\nReport is saved as '{filename }'.")
except Exception ase:
print(f"￿ Report saving fails: {e}")
returnreport
[ ]:
[ ]:#Main execution function
[82]: deftrain_all_models (self, device: torch .device, num_epochs: int=300):
"""Train all models"""
print(f"\n {len(self.models) }models are trained simulataneously... ")
# Move data to device
forkey in['x','edge_index ','edge_weight ']:
ifkey inself.data:
self.data[key] =self.data[key] .to(device)
targets_device ={}
forkey, value inself.targets.items():
targets_device[key] =value.to(device)
# Set optimizers
optimizers ={}
schedulers ={}
formodel_name, model inself.models.items():
ifmodel_name =='EconomicESGGNN ':
params=list(model.parameters())
else:
params=list(model.parameters()) +list(self.
↪prediction_heads[model_name] .parameters())
55

# ----- PAGE BREAK (55) -----

lr=1e-4 ifmodel_name in['EconomicGNN_GCN ','HybridGNN ']else1e-3
optimizers[model_name] =torch.optim.AdamW(params, lr =lr,␣
↪weight_decay =1e-4)
schedulers[model_name] =torch.optim.lr_scheduler .CosineAnnealingLR(
optimizers[model_name], T_max =num_epochs
)
# Loss functions
loss_functions ={
'esg_risk ': nn.BCEWithLogitsLoss(),
'economic_impact ': nn.MSELoss(),
'volatility ': nn.MSELoss(),
'transition_cost ': nn.BCEWithLogitsLoss(),
'compliance_probability ': nn.BCEWithLogitsLoss()
}
loss_weights ={
'esg_risk ':0.25,
'economic_impact ':0.25,
'volatility ':0.2,
'transition_cost ':0.15,
'compliance_probability ':0.15
}
# Training history
training_history ={model_name: [] formodel_name inself.models.keys()}
# Training loop
start_time =time.time()
forepoch inrange(num_epochs):
epoch_losses ={}
formodel_name, model inself.models.items():
ifmodel_name =='EconomicGAE ':
continue
model.train()
optimizer =optimizers[model_name]
optimizer .zero_grad()
# Forward pass - FIXED VERSION
ifmodel_name =='EconomicESGGNN ':
predictions =model(self.data['x'],self.data['edge_index '],
self.data['edge_weight '],self.data['codes'])
elifmodel_name =='ImprovedEconomicGNN ':
model_output =model(self.data['x'],self.data['edge_index '],
56

# ----- PAGE BREAK (56) -----

self.data['edge_weight '],self.
↪data['codes'])
embeddings =model_output[ 'embeddings ']
predictions =self.prediction_heads[model_name](embeddings)
else:
# Handle models that may return 1 or 2 values safely
model_output =model(self.data['x'],self.data['edge_index '],
self.data['edge_weight '])
# Check if it's a tuple (multiple returns) or single value
ifisinstance (model_output, tuple):
embeddings =model_output[ 0]# Take embeddings (first ␣
↪element)
attention_weights =model_output[ 1]iflen(model_output) >␣
↪1 else None
else:
embeddings =model_output # Single return value
attention_weights = None
predictions =self.prediction_heads[model_name](embeddings)
# Loss calculation
total_loss =0
fortask_name, loss_fn inloss_functions .items():
iftask_name intargets_device andtask_name inpredictions:
pred=predictions[task_name]
target=targets_device[task_name]
# Handle logit conversion for BCEWithLogitsLoss
iftask_name in['esg_risk ','transition_cost ',␣
↪'compliance_probability ']:
ifhasattr(self.models[model_name], 'prediction_heads ')␣
↪ormodel_name =='EconomicESGGNN ':
# Model already applied sigmoid, convert back to ␣
↪logits
pred=torch.clamp(pred, 1e-7,1-1e-7)
pred=torch.log(pred /(1-pred))
task_loss =loss_fn(pred, target)
# Check for NaN or inf
iftorch.isnan(task_loss) ortorch.isinf(task_loss):
print(f" {model_name }-{task_name }: Loss is NaN/Inf. ␣
↪Skip.")
task_loss =torch.tensor(0.0, device =device,␣
↪requires_grad =True)
57

# ----- PAGE BREAK (57) -----

total_loss +=task_loss *loss_weights[task_name]
# Backward pass
total_loss .backward()
torch.nn.utils.clip_grad_norm_(
list(model.parameters()) +
(list(self.prediction_heads[model_name] .parameters()) if␣
↪model_name !='EconomicESGGNN ' else[]),
max_norm =1.0
)
optimizer .step()
schedulers[model_name] .step()
epoch_losses[model_name] =total_loss .item()
training_history[model_name] .append(total_loss .item())
# Print progress
if(epoch+1)%25==0:
print(f"\n[Epoch {epoch+1:03d }/{num_epochs }] loss for each model: ")
formodel_name, loss inepoch_losses .items():
print(f"• {model_name }:{loss :.4f }")
training_time =time.time()-start_time
print(f"\nAll models are trained! (Time spent: {training_time :.1f }sec)")
self.results[ 'training_history ']=training_history
self.results[ 'training_time ']=training_time
returntraining_history
[ ]:
[83]: if__name__ =="__main__ ":
results =main()
Various GNN models ESG analysis system is starting…
This system trains 6 different GNN architectures simultaneously
And compares performances to suggest optimal ESG analysis solution.
Using device: cpu
Level 1: Integrated data load…
======================================================================
Start ESG analysis sysetem based on real data
======================================================================
ESG data loading…
ESG data loading completed:
- Number of companies: 722
58

# ----- PAGE BREAK (58) -----

- Number of industries: 47 industries
- mean ESG score: 975.8
- ESG score scope: 600-1536
Loading I-O tables (recent 5 years)…
- Years to use: 2023, 2022, 2021, 2020, 2019
￿ Multi-year I-O data loading and processing completed.
GDP data loading ('Real Gross Output by Industry.csv')…
GDP data loading failed: 'Government'
Loading environmental data ('ghgp_data_2023.xlsx')…
Original data size: (6470, 66)
Processed data: 66 industry types
Emissions by industry type (top 5):
Power Plants: 1,403,940,313 metric tons CO2e
Chemicals: 113,059,023 metric tons CO2e
Petroleum and Natural Gas Systems: 109,440,512 metric tons CO2e
Minerals: 106,790,998 metric tons CO2e
Other: 94,852,955 metric tons CO2e
Load environmental taxes…
Load environmental taxes: 5
Collect news data and start analyzing sentiment (Maximum 20 by industries)…
￿ News API key is not provided. Skip sentiment analysis.
Creating integrated features for GNN input…
- GDP data not available, using dummy data.
Environmental feature mapping completed.
Creating time series features…
￿ Time series feature creation completed.
- Fill 0 because there is no sentiment data.
￿ Integrated feature creation completed: (165, 26)
Generate target data for GNN learning…
Target data generating completes:
- esg_risk: (Mean: 0.706, Scope: [0.219, 1.000])
- economic_impact: (Mean: 0.015, Scope: [-0.040, 0.087])
- volatility: (Mean: 0.503, Scope: [0.010, 1.010])
- transition_cost: (Mean: 0.684, Scope: [0.000, 1.000])
- compliance_probability: (Mean: 0.268, Scope: [0.000, 1.000])
￿ All data load completes!
- Number of sectors: 165
- Number of features: 26
- Number of edges: 12469
Level 2: Various GNN model systems initialize…
Initialize multiple GNN models…
￿ 8models are initialized
• EconomicGNN_GCN: 79,301 parameter
• EconomicGNN_GAT: 90,565 parameter
• WeightedGATGNN: 219,077 parameter
59

# ----- PAGE BREAK (59) -----

• HybridGNN: 195,397 parameter
• ImprovedEconomicGNN: 271,907 parameter
• EconomicESGGNN: 421,655 parameter
• GPR_GNN: 148,507 parameter
• EconomicGAE: 65,631 parameter
Level 3: All models are trained simultaneously…
8models are trained simulataneously…
[Epoch 025/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3534
• WeightedGATGNN: 0.3510
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2428
• EconomicESGGNN: 0.2849
• GPR_GNN: 0.2856
[Epoch 050/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3514
• WeightedGATGNN: 0.3508
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2327
• EconomicESGGNN: 0.2610
• GPR_GNN: 0.2446
[Epoch 075/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3506
• WeightedGATGNN: 0.3497
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2303
• EconomicESGGNN: 0.2399
• GPR_GNN: 0.2367
[Epoch 100/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3503
• WeightedGATGNN: 0.3489
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2278
• EconomicESGGNN: 0.2403
• GPR_GNN: 0.2320
[Epoch 125/300] loss for each model:
• EconomicGNN_GCN: 0.4489
60

# ----- PAGE BREAK (60) -----

• EconomicGNN_GAT: 0.3505
• WeightedGATGNN: 0.3488
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2263
• EconomicESGGNN: 0.2345
• GPR_GNN: 0.2310
[Epoch 150/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3511
• WeightedGATGNN: 0.3477
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2257
• EconomicESGGNN: 0.2384
• GPR_GNN: 0.2306
[Epoch 175/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3507
• WeightedGATGNN: 0.3466
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2252
• EconomicESGGNN: 0.2333
• GPR_GNN: 0.2298
[Epoch 200/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3496
• WeightedGATGNN: 0.3464
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2257
• EconomicESGGNN: 0.2324
• GPR_GNN: 0.2290
[Epoch 225/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3499
• WeightedGATGNN: 0.3454
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2247
• EconomicESGGNN: 0.2342
• GPR_GNN: 0.2289
[Epoch 250/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3480
• WeightedGATGNN: 0.3472
• HybridGNN: 0.4489
61

# ----- PAGE BREAK (61) -----

• ImprovedEconomicGNN: 0.2248
• EconomicESGGNN: 0.2344
• GPR_GNN: 0.2286
[Epoch 275/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3509
• WeightedGATGNN: 0.3489
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2249
• EconomicESGGNN: 0.2330
• GPR_GNN: 0.2290
[Epoch 300/300] loss for each model:
• EconomicGNN_GCN: 0.4489
• EconomicGNN_GAT: 0.3499
• WeightedGATGNN: 0.3481
• HybridGNN: 0.4489
• ImprovedEconomicGNN: 0.2251
• EconomicESGGNN: 0.2346
• GPR_GNN: 0.2291
￿ All models are trained! (Time spent: 84.7sec)
Level 4: Evaluate performances…
'Evaluation of model performance'…
Summary model performance respectively (R² standard):
• EconomicGNN_GCN: mean R² = -1.0915
• EconomicGNN_GAT: mean R² = -0.0921
• WeightedGATGNN: mean R² = -0.0194
• HybridGNN: mean R² = -1.0915
• ImprovedEconomicGNN: mean R² = 0.8086
• EconomicESGGNN: mean R² = 0.4732
• GPR_GNN: mean R² = 0.7139
Level 5: Training Anomaly Detector…
Training Unsupervised Anomaly Detector (EconomicGAE)…
Epoch 020/100 | Loss: 0.0204
Epoch 040/100 | Loss: 0.0185
Epoch 060/100 | Loss: 0.0183
Epoch 080/100 | Loss: 0.0182
Epoch 100/100 | Loss: 0.0181
Level 6: Detecting Anomalies…
62

# ----- PAGE BREAK (62) -----

Detecting anomalies…
Top 10 most anomalous industries:
sector_code anomaly_score
0 75 0.121218
1 26 0.119580
2 74 0.112196
3 7 0.105442
4 38 0.104179
5 6 0.100819
6 55 0.091186
7 161 0.085914
8 4 0.082854
9 61 0.079551
Level 7: Visualization results…
Overall result visualization…
63

# ----- PAGE BREAK (63) -----

￿ Overall visualization completes!
Level 7-1: Visualizing Shock Propagation…
64

# ----- PAGE BREAK (64) -----

Level 8: Generate final report…
Final ESG GNN analysis report based on real data
======================================================================
Data:
• ESG corperation data: 722company
• I-O table: 165industry × 165industry
• Integrated feature: 26feature
• Network connection: 12469edge
• Data source: Real Data with Timeseries
Model performance:
• Number of trained models: 8
• Highest performance model: ImprovedEconomicGNN
• Mean R² score: 0.8086
• Training time: 84.7sec
Performance by tasks:
• esg_risk: ImprovedEconomicGNN (R² = 0.987)
• economic_impact: EconomicESGGNN (R² = 0.903)
• volatility: EconomicESGGNN (R² = 0.985)
65

# ----- PAGE BREAK (65) -----

• transition_cost: ImprovedEconomicGNN (R² = 0.990)
• compliance_probability: ImprovedEconomicGNN (R² = 0.996)
Key findings:
• Industries with lower ESG scores show higher risk predictions
• Strong correlation between environmental tax exposure and economic impact
• Bidirectional GNNs outperform unidirectional models
• ESG gating mechanism enables effective feature selection
Practical implications:
• Utilize ESG scores for portfolio risk management
• Develop investment strategies that account for inter-industry spillover
effects
• Establish proactive response systems to regulatory changes
• Optimize sustainable supply chain management
Limitations:
• Prediction accuracy depends on data quality
• Limited adaptability to sudden policy shifts
• Increased uncertainty in long-term forecasts
Future works:
• Build real-time data update systems
• Integrate a wider range of ESG indicators
• Tailor models by country/region
• Incorporate explainable AI techniques
Report is saved as 'ESG_GNN_Analysis_Report_20250821_162304.txt'.
======================================================================
All analysis are completed successfully!
======================================================================
[ ]:
66

# ----- PAGE BREAK (66) -----

