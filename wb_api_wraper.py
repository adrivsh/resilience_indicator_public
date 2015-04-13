from pandas.io import wb
import numpy as np


def get_wb_df(wb_name,colname):
    """gets a dataframe from wb data with all years and all countries, and a lotof nans"""    
    #return all values
    wb_raw  =(wb.download(indicator=wb_name,start=2000,end=2014,country="all"))
    #sensible name for the column
    # wb_raw.rename(columns={wb_raw.columns[0]: colname},inplace=True)
    return wb_raw.rename(columns={wb_raw.columns[0]: colname})
    
def get_wb_series(wb_name,colname='value'):
    """"gets a pandas SERIES (instead of dataframe, for convinience) from wb data with all years and all countries, and a lotof nans""" 
    return get_wb_df(wb_name,colname)[colname]
    

def get_wb_mrv(wb_name,colname):
    """most recent value from WB API"""
    return mrv(get_wb_df(wb_name,colname))
    

def mrv_gp(x):
    """this function gets the most recent value from a wb-pulled dataframe grouped by country"""
    out= x.ix[(x["year"])==np.max(x["year"]),2]
    return out    
    
def mrv(data):    
    """most recent values from a dataframe. assumes one column is called 'year'"""    
    #removes nans, and takes the most revent value. hop has a horrible shape
    hop=data.reset_index().dropna().groupby("country").apply(mrv_gp)
    #reshapes hop as simple dataframe indexed by country
    hop= hop.reset_index().drop("level_1",axis=1).set_index("country")
    return hop
    
def avg_gp(x):
    """this function gets the average over the last 10 years of a wb-pulled dataframe grouped by country"""
    last_year = float(np.max(x["year"]))
    lyten = last_year - 10;
    where = x["year"].astype(float)>lyten
    out= x.ix[where,2].mean()
    return out    
    
def avg_val(data):    
    """10 year average from a dataframe. assumes one column is called 'year'"""    
    #removes nans, and takes the most revent value. hop has a horrible shape
    hop=data.reset_index().dropna().groupby("country").apply(avg_gp)
    #reshapes hop as simple dataframe indexed by country
    hop= hop.reset_index().drop("level_1",axis=1).set_index("country")
    return hop

def get_wb_avg(wb_name,colname):
    """most recent value from WB API"""
    return avg_val(get_wb_df(wb_name,colname))
    