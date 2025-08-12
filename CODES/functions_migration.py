import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import pandas as pd
import math
import scipy 
import obspy


#################### GENERAL FUNCTIONS

def arrayToStream(array, streamStats):
    '''
    convert 2D numpy array into obspy data stream
    - streamStats is another obspy stream whose properties will be adapted

    Input: 
    array: 2D array
    streamStats: an obspy data stream

    Output:
    stream: obspy data stream

    '''

    nTr = array.shape[0]
    npts = array.shape[1]

    stream = Stream()
    for iterTr in range(nTr):
        tr = Trace(array[iterTr,:])
        tr.stats = streamStats[iterTr].stats
        stream += tr
    
    return stream


### CONVERSION GEOGRAPHIC CARTESIAN 
def LonLatToKm(coordsLon, coordsLat, coordsRefLonLat = None):
    """
    converts lon/lat into km relative to reference coordinates
    
    Input: 
    coordsLon, coordsLat: coordinates in LON/LAT (list), or simply 'IDDP1' for IDDP1 reference coordinates
    coordsRefLonLat: reference coordinates (list, e.g., [lonRef, latRef])
    
    Output: 
    returns coordinates in KM relative to coordsRefLonLat (list)
    """

    from obspy.signal.util import util_geo_km

    if coordsRefLonLat == "IDDP":
        coordsRefLonLat  = [-16.76413050, 65.71588535] # IDDP coordinates

    if coordsRefLonLat == None:
        coordsRefLonLat = [np.mean(coordsLon),np.mean(coordsLat)]

    X=[]
    Y=[]

    for iterSta in range(len(coordsLon)):
        [x,y] = util_geo_km(
                coordsRefLonLat[0],
                coordsRefLonLat[1],
                coordsLon[iterSta],
                coordsLat[iterSta],
            )
        X.append(x)
        Y.append(y)


    coords_cartesian = np.array([X,Y])
    return coords_cartesian



def getEventString(dfAll, iterQuake, config=None):
    '''
    get name of a particular event file based on pandas dataframe 'df' and the row number iterQuake 

    Input: 
    dfAll: pandas dataframe containing event info
    iterQuake: row index in dataframe (event number)

    Output: 
    Name of earthquake (string), can be used to load data
    '''

    df = dfAll.iloc()[iterQuake]

    evDate = df.Date
    evTime = df.Time
    evLat = np.round(df.Latitude, 4)
    evLon = np.round(df.Longitude, 4)
    evDepth = df.Depth
    evMag = np.round(df.Magnitude, 4)

    tmp = evTime[0:2]+evTime[3:5]+evTime[6:]
    evString = evDate+"_"+tmp + "_" + str(evLat)+"_"+str(evLon)+"_"+str(evDepth)+"_"+str(evMag)


    if config:
        evString = evString+"_"+config+".mseed"
    else:
        evString = evString+".mseed"

    return evString



#################### 1. BINNING FUNCTIONS 

def rotate_point(point, angle, refCoords = (0,0)):
    """
    Rotate a point (x, y) by a given angle in degrees.

    Input:
    point: x and y coordinate of point (x,y)

    Output: 
    rotated coordinates
    """
    xRef = refCoords[0]
    yRef = refCoords[1]
    x, y = point
    angle_rad = math.radians(angle)
    new_x = xRef + (x - xRef)*math.cos(angle_rad) - (y - yRef)*math.sin(angle_rad)
    new_y = yRef + (x - xRef)*math.sin(angle_rad) + (y - yRef)*math.cos(angle_rad)
    
    return (new_x, new_y)

def create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(0,0)):
    """
    create a rotated Cartesian grid.

    Input: 
    modelSpaceLimits: boundaries of model space (list [[xmin, xmax], [ymin, ymax]])
    grid lengths: grid increments dx and dy (list [dx, dy])
    angle: rotation angle of the grid (float)

    Output: 
    rotated_xx, rotated_yy: X- and Y grid coordinates
    """

    # Generate a regular grid
    x = np.arange(modelSpaceLimits[0][0],modelSpaceLimits[0][1],gridLengths[0])
    y = np.arange(modelSpaceLimits[1][0],modelSpaceLimits[1][1],gridLengths[1])
    xx, yy = np.meshgrid(x, y)
    
    # Rotate each point in the grid
    rotated_xx, rotated_yy = rotate_point((xx, yy), angle, refCoords=refCoords)
    
    return rotated_xx, rotated_yy



def compute_center_point(x1, y1, x2, y2, x3, y3, x4, y4):
    '''
    compute center point within a grid cell defined by corners

    Input: 
    x1, y1, x2, y2, x3, y3, x4, y4: bounds of cell (float)

    Output: 
    center_x, center_y: center coordinates of grid cell
    '''
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4
    return (center_x, center_y)




def extract_values_within_grid(dataframe, grid_vertices, x_col, y_col):
    from shapely.geometry import Point, Polygon
    
    """
    Extracts values from a DataFrame that fall within a specified grid cell.
    
    Input:
    dataframe: pandas dataFrame with x and y coordinates
    grid_vertices (list of tuples): list of (x, y) tuples representing the vertices of the grid cell
    x_col: column name for the x-coordinate in the DataFrame (str)
    y_col: column name for the y-coordinate in the DataFrame (str)

    Output:
    pd.DataFrame: dataFrame containing only the rows where (x, y) falls within the grid cell.
    """
    # Create a Polygon for the grid cell
    grid_polygon = Polygon(grid_vertices)
    
    # Function to check if a point is within the polygon
    def point_in_grid(row):
        point = Point(row[x_col], row[y_col])
        return grid_polygon.contains(point)
    
    # Filter rows where points fall within the grid cell
    filtered_df = dataframe[dataframe.apply(point_in_grid, axis=1)]
    
    return filtered_df

def keep_columns_by_name(dataframe, substring):
    """
    creates sub-DataFrame which retains only columns where the column name contains a given substring
    
    Input:
    dataframe: The original DataFrame to filter (pd.DataFrame).
    substring: The substring to search for within the column names (str).

    Output:
    pd.DataFrame: dataFrame with only the columns whose names contain the substring
    """
    # Identify columns that contain the substring in their names
    matching_columns = [col for col in dataframe.columns if substring in col or col == 'Reflector_depth']
    
    # Create a sub-DataFrame with only the matching columns
    filtered_df = dataframe[matching_columns]
    
    return filtered_df

def combine_dataframes(dataframes, key):
    """
    combine multiple DataFrames into a single DataFrame.
    - stacks DataFrames horizontally if 'Reflector depth' values match.
    - stacks DataFrames vertically otherwise.
    -fills any resulting empty cells with zeros.

    Input:
    - dataframes: dataFrames to combine (list of pd.DataFrame)

    Output:
    combined DataFrame with empty cells filled with zeros (pd.DataFrame)
    """

    # Start with the first DataFrame
    combined_df = dataframes[0]
    
    for df in dataframes[1:]:
        # Check if 'Reflector depth' values match
        if combined_df[key].equals(df[key]):
            # If they match, join horizontally by index
            combined_df = pd.concat([combined_df, df.drop(columns=[key])], axis=1)
        else:
            # If they don't match, stack vertically
            combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)
    
    # Fill NaN values with zeros
    combined_df = combined_df.fillna(0)
    
    return combined_df




################# PRE-PROCESSING FUNCTIONS

def returnReflectorDepths(wavetype):
    '''
    extract reflector depth values used in Raytracing algorithm. 
    - needs to be adapted based on chosen dzd increment in Raytracing algorithm

    Input:
    wavetype: 'R*' for primary reflections (e.g., RP), and 'G*' for multiple reflections (e.g., 'GP')

    Output: 
    reflector depth values 
    '''

    # reflector depths for multiples
    if wavetype.startswith('G'):
        dzd = 0.05
        reflectorDepthsAll= np.arange(0.57, 7 + dzd, dzd)

    # reflector depths for primary reflections
    elif wavetype.startswith('R'):
        dzd = 0.02
        startval = 0
        reflectorDepthsAll= np.arange(startval, 11 + dzd, dzd)

    reflectorDepthsAll = [float("{:.2f}".format(num)) for num in reflectorDepthsAll]

    return reflectorDepthsAll


def get_timeLag(array1, array2):

        ''' 
        get optimal time shift between two arrays based on cross-correlation

        Input: 
        array1: first data vector (np.array)
        arr2: second data vector (np.array)

        Output: 
        optimum time lag in samples
        '''

        corr = scipy.signal.correlate(array1, array2, mode='same', method='auto')

        ### index in cross-correlation corresponding to maximum amplitude
        maxIdx = np.argmax(corr)
        timeLagInSamples = maxIdx - len(corr)//2

        return timeLagInSamples



def compute_psd(time_series, sampling_freq):
    import scipy.fft as ft

    """
    power spectral density (PSD) of a time series using scipy
    
    Input: 
    time_series: time series data (np.array)
    sampling_freq: sampling rate of the time series.
        
    Output:
    freqs: frequencies corresponding to the PSD (array)
    psd (ndarray): power spectral density (array)
    """

    # FFT
    fft_vals = ft.fft(time_series)
    n = len(time_series)
    freqs = ft.fftfreq(n, 1 / sampling_freq)
    
    # Compute the one-sided spectrum
    positive_freqs = freqs[:n//2]
    psd = (1 / (sampling_freq * n)) * np.abs(fft_vals[:n//2])**2
    
    return positive_freqs, psd



def apply_maxAmp(stream, maxAmpWin, tt_DP, method = 'max'):
    ''' 
    extract maximum amplitude within a pre-defined time window and shift trace such that maximum 
    amplitude is aligned with specific traveltime
    - procedure is carried out for each trace ins tream separately

    Input: 
    stream: data to be shifted (obspy data stream)
    maxAmpWin: length of time window to be included 
    tt_DP: traveltimes to which maximum amplitude will be shifted

    Output: 
    shifted stream
    '''

    import copy
    stream_mod = copy.deepcopy(stream)


    sampRate = stream_mod[0].stats.sampling_rate
    for trace, iterTr in zip(stream_mod, range(len(stream))):
        sTime = trace.stats.starttime
        endTime = trace.stats.endtime
        Pwin_sig = copy.deepcopy(trace)
        Pwin_sig.trim(sTime + tt_DP[iterTr] - maxAmpWin/2, sTime + tt_DP[iterTr] + maxAmpWin/2, pad = True, fill_value = 0)
        idxMax = np.argmax(Pwin_sig.data)
        if method == 'absmax':
            idxMax = np.argmax(np.abs(Pwin_sig.data))

        idxMaxSec = idxMax/sampRate
        idxMaxSecA = tt_DP[iterTr] - maxAmpWin/2 + idxMaxSec

        shiftBy = tt_DP[iterTr] - idxMaxSecA
        trace.trim(sTime - shiftBy, endTime - shiftBy, pad = True, fill_value = 0)
   
        trace.stats.starttime = sTime
    
    return stream_mod


def get_traveltimes(index, stations):
    ''' 
    extract traveltimes for direct P and S waves from file

    Input: 
    index: index of event for ehcih traveltimes will be extracted (NOT row number)
    stations: stations for which traveltimes will be extracted

    Output: 
    tt_DS, tt_DP: traveltimes for direct P and S waves for all stations (arrays)
    '''

    path_file = '../META/PStraveltimes' 
    ttimes = pd.read_csv(path_file+".csv")
    earthquake = ttimes.loc()[index]
    Nrec = len(stations)

    tt_DP = np.zeros(Nrec)
    tt_DS = np.zeros(Nrec)


    for iter in range(Nrec):
        station = stations[iter] 
        if station.startswith('ARR'):
            station = 'ARR0' + station[-2:]
        tt_DP[iter] = (earthquake[station + "_tt_dP[s]"])
        tt_DS[iter] = (earthquake[station + "_tt_dS[s]"])

    return tt_DP, tt_DS

def preprocess_data(stream, iterQuake, freqRange, maskDirectWaves = True, normalize_stream = True):

    ''' 
    Data preprocessing function

    Input: 
    stream: obspy data stream
    iterQuake: event number to be analysed (int)
    freqRange: frequency range (list [freqmin, freqmax]) - a bandpass filter will be applied
    maskDirectWaves: if True, direct P- and S waves as well as S wave coda will be masked by setting amplitudes to 0
    normalize_stream: if True, each trace will be normalized by its rms between P and S direct wave arrivals

    Output: 
    Preprocessed obspy data stream
    '''

    # quick check of nans in trace
    for tr in stream: 
        if np.isnan(tr.data).any():
            # Replace NaNs with 0
            tr.data = np.nan_to_num(tr.data, nan=0.0)

    # tapering and detrending
    stream.taper(0.05, side = 'both')
    stream.detrend('linear')
    stream.detrend('constant')

    # frequency filtering
    freqmin = freqRange[0]
    freqmax = freqRange[1]

    if not (freqmin == None) or (freqmax == None):
        stream.filter('bandpass', freqmin = freqmin, freqmax = freqmax, zerophase = True, corners = 1)
    
    # obtain traveltimes of direct P and S waves for masking and normalization
    stations = [tr.stats.station for tr in stream]
    tt_DP, tt_DS = get_traveltimes(iterQuake, stations)
    
    corrFacs = tt_DP - 0.5
    tt_DP -= corrFacs
    tt_DS -= corrFacs

    # mask direct waves
    if maskDirectWaves:
        for tr, iterTr in zip(stream, range(len(stream))):
            ### mute P and S waves
            sampRate = tr.stats.sampling_rate
            tt_DP_samp = np.round(tt_DP[iterTr]*sampRate)
            tt_DS_samp = np.round(tt_DS[iterTr]*sampRate)
            Pwin_delete_samp = 6
            tr.data[0:int(tt_DP_samp+Pwin_delete_samp)] = 0

            ### mute S wave and S coda
            tr.data[int(tt_DS_samp):] = 0

    # data normalization
    if normalize_stream:
        sampRate = stream[0].stats.sampling_rate
        for tr, iterTr in zip(stream, range(len(stream))):
            tt_DP_samp = np.round(tt_DP[iterTr]*sampRate)
            tt_DS_samp = np.round(tt_DS[iterTr]*sampRate)
            data_chunk = tr.data[int(tt_DP_samp):int(tt_DS_samp)]
            rms = np.sqrt(np.mean(data_chunk ** 2))
            if rms > 0:
                tr.data /= rms
    return stream


def check_for_combi_file(iterQuake, df, pathCombis):
        ''' 
        return csv file which contains the binning results (output of 1_binning.py)

        Input: 
        iterQuake: event number to be analysed (int)
        df: table with event info (pd.DataFrame)
        pathCombis: path where CSV-binning files are stored

        Output: 
        binCombisEvents: the csv file (pd.DataFrame)
        '''

        evString = getEventString(df, iterQuake)
        evString = evString.split('.mseed')[0]
        fNameCombisEvent = pathCombis + evString + '.csv'
    
        binCombisEvent = pd.read_csv(fNameCombisEvent)
        return binCombisEvent


########### PLOTTING FUNCTIONS 

def local_coherency_filter(data,
                           x_positions,
                           slopes,
                           aperture_depth=11,
                           aperture_width=7,
                           semblance=True):

    from scipy.ndimage import shift

    """
    Apply local coherence-based weighting to each sample in the input data.

    Parameters:
        data : 2D numpy array (depth x x)
        x_positions : 1D array of x positions (len = number of columns in data)
        slopes : list or array of slopes to test (samples per x)
        aperture_depth : number of depth samples in window (must be odd)
        aperture_width : number of x samples in window (must be odd)
        semblance : use semblance (True) or energy (False)

    Returns:
        output : 2D array (same shape as data) with values scaled by local coherence
    """
    n_z, n_x = data.shape
    output = np.zeros_like(data)

    half_d = aperture_depth // 2
    half_w = aperture_width // 2

    for ix in range(n_x):
        # Determine available traces on left and right
        left = max(0, ix - half_w)
        right = min(n_x, ix + half_w + 1)
        actual_aperture = right - left

        for iz in range(half_d, n_z - half_d):
            # Extract depth window
            window = data[iz - half_d:iz + half_d + 1, left:right]
            x_win = x_positions[left:right]

            best_score = -np.inf

            for slope in slopes:
                aligned = np.zeros_like(window)

                for i, x in enumerate(x_win):
                    shift_amt = -slope * (x - x_positions[ix])
                    aligned[:, i] = shift(window[:, i], shift=shift_amt, order=1, mode='nearest')

                if semblance:
                    # Semblance computation
                    numerator = np.sum(np.sum(aligned, axis=1)**2)
                    denominator = aligned.shape[1] * np.sum(aligned**2)
                    score = numerator / (denominator + 1e-10)
                else:
                    # Energy-based coherency
                    stack = aligned.mean(axis=1)
                    score = np.sum(stack**2)

                if score > best_score:
                    best_score = score

            # Multiply center sample by max coherency
            output[iz, ix] = data[iz, ix] * best_score

    return output
