import pandas as pd
import numpy as np
import sys
          
from datetime import datetime
def datestring_to_daynum(datestring):
    "given datestring 01/01/2020, return number of days since linux epoch"
    t = datetime.strptime(datestring, '%Y-%m-%d')
    
    #datenum = t.timetuple().tm_yday
    daynum = (t - datetime(1970,1,1)).days
    #print(f"{datestring} -> {daynum}")
    return daynum

# convert DATESTRING to proper timestamp
def datestring_to_timestamp(datestring,format='%Y-%m-%d'):
    #print(f"datestring_to_timestamp({datestring})")
    return pd.to_datetime(datestring, format=format)

def calc_delta_over_daynum(df, county, state, srccol, destcol, floorval=0.0):
  """
  given a dataframe df and the locale, 
  compute dcases = deltacases = new cases daily
  returns updated df

  TODO: replace w/ rolling
  """
  assert "county" in df
  assert "state" in df
  # add new column
  if destcol not in df:
    df[destcol] = 0.0
  #end

  indices= (df.county==county) & (df.state==state)
  tmp_df = df[indices].sort_values('daynum',ascending=True)
  assert(len(tmp_df)>0)
  x = tmp_df[srccol]
  h = np.array([+1.0,-1.0])
  delta = np.convolve(h,x,mode='same')
  #print(cases)
  #print(f"delta={delta}")

  # eliminate values < 0 because # cases should be monotonic
  floorvec = floorval * np.ones(delta.shape,dtype='float')
  assert(floorvec.shape==delta.shape)
  deltamax=np.maximum(floorvec, delta)
  #tmp_df[destcol] = deltamax

  # assign nd values back into df 
  df.loc[indices,destcol] = deltamax
  return df

def scale_col_values(df, county, state, scalef, srccol, destcol):
  """
  given a dataframe df and the locale, 
  compute dcases = deltacases = new cases daily
  returns updated df

  TODO: replace w/ rolling
  """
  assert "county" in df
  assert "state" in df
  assert srccol in df
  assert scalef >= 0.0
  # add new column
  if destcol not in df:
    df[destcol] = 0.0
  #end

  indices= (df.county==county) & (df.state==state)
  tmp_df = df.loc[indices,srccol]
  assert(len(tmp_df)>0)
  
  x = df.loc[indices,srccol]
  y = scalef * x

  # assign nd values back into df 
  df.loc[indices,destcol] = y
  return df


import sys
def calc_growthfactor(df, county, state,destcol='growthfactor'):
    """
    given county + state,
    calc growth factor per day, write to column 'growthfactor'
    """
    MIN_FLOAT=1.0e-10
    if destcol not in df:
        # add new column
        df[destcol] = 0.0
    #end

    indices = (df.county==county) & (df.state==state)
    tmp_df = df[indices].sort_values('daynum',ascending=True)
    print(tmp_df.tail())
    print(len(tmp_df))
    assert(len(tmp_df)>0)
    start_daynum = tmp_df['daynum'].min()
    stop_daynum = tmp_df['daynum'].max()
    daynum_list=np.arange(start_daynum+1, stop_daynum+1)
    print(daynum_list)
    assert len(daynum_list)==len(tmp_df), f"len(daynum_list)={len(daynum_list)}, len(tmp_df) = {len(tmp_df)}"
    
    gf_list=[0.0]  # the first growthf value
    for daynum in daynum_list:
       dn = tmp_df[tmp_df.daynum==daynum].dcases.values[0]
       dn1 = tmp_df[tmp_df.daynum==daynum-1].dcases.values[0]
       #print(f"dn={dn}, dn1={dn1}")
       gf = float(dn)/( float(dn1) + MIN_FLOAT)
       gf_list.append( gf )
    #end
    #tmp_df["growthf"] = gf_list
    #print(tmp_df.head())

    # assign nd values back into df by overwriting rows
    # print(f"len(df.loc) = {len(df.loc[indices,destcol])}")
    print(f"{len(df.loc[indices,destcol])} =?= {len(gf_list)}")
    assert len(df.loc[indices,destcol]) == len(gf_list)
    # print(f"len(daynum_list) = {len(daynum_list)}")
    
    df.loc[indices,destcol] = gf_list
    return df

# def avg_over_daynum(df,county, state, srccol, numpoints):
#   """
#   average srccol over daynum, return series
#   """
#   assert srccol in df
  
#   indices = (df.county==county) & (df.state==state)
  
#   tmp_df = df[indices].sort_values('daynum',ascending=True)
#   assert(len(tmp_df)>0)
#   #print(tmp_df.tail())
#   x = tmp_df[srccol]
#   h = np.array([1.0/numpoints] * numpoints)

#   filtered = np.convolve(h,x,mode='same')

#   return filtered

def split_county_state(df):
    """split county+state into two different columns
    drop county+state column
    return new data frame
    """
    cs_s = df["county+state"]  # a series
    print(len(cs_s))
    
    # each county+state entry has form "county,state"
    # split into list of [ [county, state], ...]
    split_list = [x.split(',') for x in cs_s]
    
    # split list into county and state series
    county_s = pd.Series([x[0] for x in split_list])
    state_s = pd.Series([x[1] for x in split_list])
    print(len(state_s))    
    ret_df = df.drop(columns=["county+state"])
    ret_df['county'] = county_s
    ret_df['state'] = state_s
    return ret_df

def strip_leading_dot(df,colname):
    """get rid of the leading "." in column entries
    """
    loc_s = df[colname] # a series
    
    # fix county names
    fixed_list = []
    for s in loc_s:  # s is a string, apparently
        if s[0]=='.':
            fixed_list.append(s[1:])
        else:
            fixed_list.append(s)
        #end
    #end
    df[colname] = pd.Series(fixed_list)
    return df

def strip_spaces(df, colname):
    x_s = df[colname]
    y_s = pd.Series([x.strip() for x in x_s])
    df[colname] = y_s
    return df
        
def to_lower(df, colname):
    x_s = df[colname] # a series
    y_s = pd.Series([x.lower() for x in x_s])
    df[colname] = y_s
    return df

def strip_trailing_county(df,colname):
    """get rid of the word 'county'
    """
    loc_s = df[colname]
    # fix county names
    fixed_list = []
    for s in loc_s:  # s is a string, apparently
        result = s.find('county')
        if result >= 0:
            fixed_list.append(s[0:result])
        else:
            fixed_list.append(s)
        #end
    #end
    df[colname] = pd.Series(fixed_list)
    return df

def do_misc_nyt_fixup(df):
    ret_df = df.copy()

    # 4/6/2020 entries: 'new york city' is entered as 'new york'
    indices=(df.county=='new york') & (df.county=='new york')
    ret_df.loc[indices,'county'] = 'new york city'
    return ret_df
    
def do_misc_census_fixup(df):
    """
    do miscellaneous fix of census data
    """
    ret_df = df.copy()
    #
    # 'new york county' -> 'new york city'
    indices = (df.county.str.contains('new york')) & (df.state=='new york')
    ret_df.loc[indices,'county'] = 'new york city'

    # kansas city falls into multiple counties!
    # add a new row:  'kansas city' county with population data data from wikipedia
    tmp_df = pd.DataFrame({"county":["kansas city"],
                           "state":["missouri"],
                           2019: [491918]})
    ret_df = pd.concat([ret_df, tmp_df],sort=False)

    # joplin, MO falls into multiple counties!
    # add a new row:  'joplin' county with population data data from internets
    tmp_df = pd.DataFrame({"county":["joplin"],
                           "state":["missouri"],
                           2018: [50657]})
    ret_df = pd.concat([ret_df, tmp_df],sort=False)

    return ret_df

def get_index_county_state(census_df, county, state, smartmatch=True):
    """
    return the indices containing 'county' and 'state',
    handling the vagaries of the county naming convention
    """
    if smartmatch==True:
        # county:
        #   match county + suffix, e.g. "anchorage" <-> "anchorage municipality"
        #   match county exactly, e.g.
        #m = ((census_df.county==county) | census_df.county.str.contains(county + " county") | census_df.county.str.contains(county + " parish") | census_df.county.str.contains(county + " municipality")) & (census_df.state==state)
        m = ((census_df.county==county) | (census_df.county==(county + " county")) | (census_df.county==(county + " parish")) | (census_df.county==(county + " municipality"))) & (census_df.state==state)
    else:
        m = (census_df.county.str.contains(county)) & (census_df.state==state)
        
    return m

# validation: check that each county+state in nyt data has population data
def validate_county_match(nyt_df, census_df):
    # generate list of all counties and states in NYT data
    sc_df = nyt_df[["county","state"]]
    sc_df = sc_df.drop_duplicates()
    assert len(sc_df) > 0, "sc_df contains no county/state pairs"
    #print(len(sc_df))
    #sc_df = pd.DataFrame({"county":["bexar"],"state":"texas",}) # 
    #print(sc_df.tail())
    #return

    # preallocate mismatch_list to avoid append
    mismatch_list=[None] * len(census_df.index)
    list_index = 0
    
    #sc_df.loc[(sc_df['county']=='asotin') & (sc_df['state']=='washington')]
    for index,sc_row_df in sc_df.iterrows():
        county=sc_row_df['county']
        state=sc_row_df['state']
        #print("-----")
        #print(f"searching for {county},{state}")
        #print(census_df[(census_df.county.str.contains(county) )])
        probe=census_df[get_index_county_state(census_df,county,state)]
        #print(f"len probe={len(probe)}")
        #print(f"{probe}")
        #print(f"census_df {county},{state}={census_df.loc[census_df['county']==county]}")
        if len(probe) <= 0:
            #print(f"could not find population data for {county},{state}")
            #mismatch_list.append((county,state))
            mismatch_list[list_index]=(county,state)
            list_index += 1
        elif len(probe) > 1:
            #print(f"found multiple ({len(probe)}) matches for {county},{state}")
            mismatch_list[list_index] = (county,state)
            #mismatch_list.append((county,state))
            list_index += 1
        else:
            assert len(probe)==1
        #end
    #end            
    return mismatch_list[0:list_index]

def fix_date(nyt_df):
    """
    convert datestring column to timestamp, 'tstamp'
    convert datestring column to daynum, 'daynum'
    """
    ret_df = nyt_df.copy()

    #
    # add new columns
    ret_df['tstamp'] = 0.0
    ret_df['daynum'] = 0.0
    
    date_list = ret_df['date'].unique()

    count = 0
    for date in date_list:
        ts  = datestring_to_timestamp(date)
        daynum = datestring_to_daynum(date)
        #print(daynum)
        ret_df.loc[ret_df.date==date, 'daynum']=daynum   # NOTE: ret_df[ret_df.date==date, 'daynum'] does NOT work; need to use .loc[]
        ret_df.loc[ret_df.date==date, 'tstamp']=ts

        if (count % 20)==0:
            print(f"{count} ",end='');sys.stdout.flush()
        #end
        count += 1
    #end
    print("")
    return ret_df


        
def filter_rows_by_state_county(df, cs_list):
    """
    given dataframe and list of (county,state) tuples,
    return a dataframe with just the rows containing (county, state)
    """
    ret_df = pd.DataFrame()
    for county, state in cs_list:
        sub_df = df.loc[(df.county==county) & (df.state==state)]
        assert len(sub_df) > 0, f"could not find any rows for ({county},{state})"
        ret_df = pd.concat([ret_df,sub_df])
                           
    #end
    return ret_df

def normalize_cases_deaths(nyt_df, pop_df, norm_cases_name='norm_cases',norm_deaths_name='norm_deaths'):
    """
    given nyt and population (census) dataframes,
    normalize cases by population -> norm_cases
    and deaths -> norm_deaths

    return dataframe with extra columns "norm_cases" and "norm_deaths"
    """
    ret_df = nyt_df.copy()
    if norm_cases_name not in ret_df:
        ret_df[norm_cases_name] = 0.0
    if norm_deaths_name not in ret_df:
        ret_df[norm_deaths_name] = 00
        
    
    for index,row in ret_df.iterrows():
        county=row['county']
        state=row['state']
        # get population for (county, state)
        try:
            pop_index = get_index_county_state(pop_df, county,state)
            pop=1.0 * pop_df[pop_index][2019].values[0] 

            deaths = 1.0 * row['deaths']
            cases = 1.0 * row['cases']
            ret_df.loc[index,norm_deaths_name] = deaths/pop
            ret_df.loc[index,norm_cases_name] = cases/pop
        except:
            #
            # exception  occurs when population data not available
            ret_df.loc[index, norm_deaths_name] = -1.0
            ret_df.loc[index,norm_cases_name] = -1.0
        #end
    #end
    print(f"created column {norm_deaths_name}")
    print(f"created column {norm_cases_name}")
    return ret_df

import pickle
def pickle_dataframes(nyt_df, nyt_fname, pop_df, pop_fname):
    # save off data
    #
    # save off data at this point
    with open(nyt_fname,mode='wb') as tmpfile:
        pickle.dump(nyt_df, tmpfile)
    #end

    with open(pop_fname,mode='wb') as tmpfile:
        pickle.dump(pop_df, tmpfile)
    #end
    return

def unpickle_dataframes(nyt_fname, pop_fname):
    with open(nyt_fname,'rb') as tmpfile:
        nyt_df = pickle.load(tmpfile)
    with open(pop_fname,'rb') as tmpfile:
        pop_df = pickle.load(tmpfile)
    #end

    return nyt_df, pop_df
