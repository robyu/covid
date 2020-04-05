import pandas as pd

from datetime import datetime
def datestring_to_daynum(datestring):
    "given datestring 01/01/2020, return the day number (of year)"
    t = datetime.strptime(datestring, '%Y-%m-%d')
    datenum = t.timetuple().tm_yday
    return datenum

# convert DATESTRING to proper timestamp
def datestring_to_timestamp(datestring,format='%Y-%m-%d'):
   return pd.to_datetime(datestring, format=format)

def calc_delta_over_daynum(df, locale, srccol, destcol, floorval=0.0):
  """
  given a dataframe df and the locale, 
  compute dcases = deltacases = new cases daily
  returns updated df

  TODO: replace w/ rolling
  """
  assert srccol in df
  # add new column
  if destcol not in df:
    df[destcol] = 0.0
  #end

  tmp_df = df[df.county==locale].sort_values('daynum',ascending=True)
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
  tmp_df[destcol] = deltamax

  # assign nd values back into df by overwriting rows
  df.loc[df['county']==locale,:] = tmp_df
  return df

import sys
def calc_growthfactor(df, locale):
  """
  calc growth factor per day
  """
  MIN_FLOAT=1.0e-10
  if 'growthf' not in df:
    # add new column
    df['growthf'] = 0.0
  #end

  tmp_df = df[df.county==locale].sort_values('daynum',ascending=True)
  #print(tmp_df.tail())
  assert(len(tmp_df)>0)
  start_daynum = tmp_df['daynum'].min()
  stop_daynum = tmp_df['daynum'].max()
  daynum_list=np.arange(start_daynum+1, stop_daynum+1)
  gf_list=[0.0]  # the first growthf value
  for daynum in daynum_list:
    dn = tmp_df[tmp_df.daynum==daynum].dcases
    dn1 = tmp_df[tmp_df.daynum==daynum-1].dcases
    #gf =sys.float_info.min
    gf = float(dn)/( float(dn1) + MIN_FLOAT)
    gf_list.append( gf )
    #print(f"day {daynum} dn={float(dn)}/dn1={float(dn1)} + {MIN_FLOAT}-> gf={gf}")
  #end
  tmp_df["growthf"] = gf_list
  #print(tmp_df.head())

  # assign nd values back into df by overwriting rows
  df.loc[df.county==locale,:] = tmp_df
  return df

def avg_over_daynum(df,locale, srccol, numpoints, destcol):
  """
  average srccol over daynum, write result to destcol
  """
  assert srccol in df
  if destcol not in df:
    df[destcol] = 0.0
  #end
  tmp_df = df[df.county==locale].sort_values('daynum',ascending=True)
  assert(len(tmp_df)>0)
  #print(tmp_df.tail())
  x = tmp_df[srccol]
  h = np.array([1.0/numpoints] * numpoints)

  filtered = np.convolve(h,x,mode='same')

  tmp_df[destcol] = filtered
  #print(tmp_df.tail())
  # assign nd values back into df by overwriting rows
  df.loc[df['county']==locale,:] = tmp_df
  return df

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
    #ret_df = ret_df.reset_index()
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
    
    mismatch_list=[]
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
            print(f"could not find population data for {county},{state}")
            mismatch_list.append((county,state))
        elif len(probe) > 1:
            print(f"found multiple ({len(probe)}) matches for {county},{state}")
            mismatch_list.append((county,state))
        else:
            assert len(probe)==1
        #end
    #end            
    return mismatch_list
