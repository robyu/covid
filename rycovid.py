print("rycovid.py 3")

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

