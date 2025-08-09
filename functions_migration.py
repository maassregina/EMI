import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
# from obspy import read, Stream, Trace, read_inventory
# from obspy.core import UTCDateTime
import pandas as pd
import math
import scipy 
import obspy


#################### GENERAL FUNCTIONS

def arrayToStream(array, streamStats):
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
    ### everything in LON/LAT
    ### returns coords in KM

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
    """
    xRef = refCoords[0]
    yRef = refCoords[1]
    x, y = point
    angle_rad = math.radians(angle)
    new_x = xRef + (x - xRef)*math.cos(angle_rad) - (y - yRef)*math.sin(angle_rad)
    new_y = yRef + (x - xRef)*math.sin(angle_rad) + (y - yRef)*math.cos(angle_rad)
    
    # new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    # new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    


    return (new_x, new_y)

def create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(0,0)):
    """
    Create a rotated Cartesian grid.
    """
    # Generate a regular grid
    x = np.arange(modelSpaceLimits[0][0],modelSpaceLimits[0][1],gridLengths[0])
    y = np.arange(modelSpaceLimits[1][0],modelSpaceLimits[1][1],gridLengths[1])
    xx, yy = np.meshgrid(x, y)
    
    # Rotate each point in the grid
    rotated_xx, rotated_yy = rotate_point((xx, yy), angle, refCoords=refCoords)
    
    return rotated_xx, rotated_yy



def compute_center_point(x1, y1, x2, y2, x3, y3, x4, y4):
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4
    return (center_x, center_y)




def extract_values_within_grid(dataframe, grid_vertices, x_col, y_col):
    from shapely.geometry import Point, Polygon
    
    """
    Extracts values from a DataFrame that fall within a specified grid cell.
    
    Parameters:
    - dataframe (pd.DataFrame): DataFrame with x and y coordinates.
    - grid_vertices (list of tuples): List of (x, y) tuples representing the vertices of the grid cell.
    - x_col (str): Column name for the x-coordinate in the DataFrame.
    - y_col (str): Column name for the y-coordinate in the DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing only the rows where (x, y) falls within the grid cell.
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
    Creates a sub-DataFrame by retaining only columns where the column name contains a given substring.
    
    Parameters:
    - dataframe (pd.DataFrame): The original DataFrame to filter.
    - substring (str): The substring to search for within the column names.

    Returns:
    - pd.DataFrame: A DataFrame with only the columns whose names contain the substring.
    """
    # Identify columns that contain the substring in their names
    matching_columns = [col for col in dataframe.columns if substring in col or col == 'Reflector_depth']
    
    # Create a sub-DataFrame with only the matching columns
    filtered_df = dataframe[matching_columns]
    
    return filtered_df

def combine_dataframes(dataframes, key):
    """
    Combines multiple DataFrames into a single DataFrame.
    - Stacks DataFrames horizontally if 'Reflector depth' values match.
    - Stacks DataFrames vertically otherwise.
    - Fills any resulting empty cells with zeros.

    Parameters:
    - dataframes (list of pd.DataFrame): List of DataFrames to combine.

    Returns:
    - pd.DataFrame: The combined DataFrame with empty cells filled with zeros.
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

    if wavetype.startswith('G'):
        dzd = 0.05
        reflectorDepthsAll= np.arange(0.57, 7 + dzd, dzd)

    elif wavetype.startswith('R'):
        dzd = 0.02
        startval = 0
        reflectorDepthsAll= np.arange(startval, 11 + dzd, dzd)

    reflectorDepthsAll = [float("{:.2f}".format(num)) for num in reflectorDepthsAll]

    return reflectorDepthsAll


def get_timeLag(array1, array2):

        corr = scipy.signal.correlate(array1, array2, mode='same', method='auto')
        maxIdx = np.argmax(corr)
        timeLagInSamples = maxIdx - len(corr)//2

        return timeLagInSamples



def compute_psd(time_series, sampling_freq):
    import scipy.fft as ft

    """
    Compute the Power Spectral Density (PSD) of a time series using FFT.
    
    Parameters:
        time_series (array_like): The time series data.
        sampling_freq (float): The sampling frequency of the time series.
        
    Returns:
        freqs (ndarray): The frequencies corresponding to the PSD.
        psd (ndarray): The power spectral density.
    """
    # Perform FFT
    fft_vals = ft.fft(time_series)
    n = len(time_series)
    freqs = ft.fftfreq(n, 1 / sampling_freq)
    
    # Compute the one-sided spectrum
    positive_freqs = freqs[:n//2]
    psd = (1 / (sampling_freq * n)) * np.abs(fft_vals[:n//2])**2
    
    return positive_freqs, psd



def apply_xcorr_wavelet(stream_aligned, tt_DP, plotShifting = False, trimLenforXcorr = None):
    from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
    stNames = [tr.stats.station for tr in stream_aligned]
    tLagsInSec = []
    normFacs = []

    for iterStation in range(len(stream_aligned)):
        station = stNames[iterStation]
        tt_DP_stat = tt_DP[iterStation]
        trace = stream_aligned.select(station = station)[0]
        if np.all(trace.data == 0):
            continue
    # traceOri = streamOri.select(station = station)[0]

        
    # if iterStation in np.arange(0, 33, 1):
        centFreq = 0
        trimLenForPSD_beg = 0.05

        trimLenForPSD = trimLenForPSD_beg

        while centFreq == 0: 
        #  print(trimLenForPSD)
            trimLenHalf = trimLenForPSD/2

            traceTrim = copy.deepcopy(trace)
            sTime = traceTrim.stats.starttime
            traceTrim.trim(sTime + tt_DP_stat - trimLenHalf, sTime + tt_DP_stat + trimLenHalf)
            
            maxiTwin = np.max(abs(traceTrim.data))
            sampFreq = traceTrim.stats.sampling_rate
            
            ## smoothing 
            frqs, psd = compute_psd(traceTrim.data, sampFreq)
            psd = konno_ohmachi_smoothing(psd, frqs, bandwidth=40, count=1, enforce_no_matrix=False, max_memory_usage=512, normalize=True)
            
            ## extract central frequency 
            centFreq = frqs[np.argmax(psd)]

            ## in case centre frequency is 0 
            trimLenForPSD = trimLenForPSD + 0.01


        ## create wavelet at given traveltime with calculated centre frequency 
        t = trace.times()
        traveltime = tt_DP_stat
        wavelet_duration = 1/centFreq *1
        #wavelet = np.sin(2 * np.pi * centFreq * (t - traveltime)) * np.exp(-((t - traveltime) ** 2) / (2 * (wavelet_duration / 6) ** 2))
        centFreq = 10
        wavelet = np.sin(2 * np.pi * centFreq * (t - traveltime)) * np.exp(-((t - (traveltime + wavelet_duration / 2))**2) / (2 * (wavelet_duration / 6) ** 2))
        
        ## prepare wavelet and data for cross-correlation 
        if not trimLenforXcorr:
            trimLenforXcorr = 1/centFreq*2
        trimLenHalf = trimLenforXcorr/2        
        traceXcorr = copy.deepcopy(trace)
        traceXcorr.trim(sTime + tt_DP_stat - trimLenHalf, sTime + tt_DP_stat + trimLenHalf)
        #traceXcorr.trim(sTime + tt_DP_stat, sTime + tt_DP_stat + trimLenHalf)

        traceXcorr.taper(0.1)

        # make sure they are the same length 
        endTime = trace.stats.endtime
        traceXcorr.trim(sTime, endTime, pad = True, fill_value = 0 )


        wavelet /= max(abs(wavelet))

        # now cross-correlate 
        tLag = get_timeLag(traceXcorr.data/max(abs(traceXcorr.data)), wavelet)
        tLagInSec = tLag/sampFreq

        ## shift Trace to correct traveltime
        traceShifted = copy.deepcopy(trace)
        traceShifted.trim(sTime + tLagInSec, endTime + tLagInSec, pad = True, fill_value = 0)

        # traceShiftedOri = copy.deepcopy(traceOri)
        # traceShiftedOri.trim(sTime + tLagInSec, endTime + tLagInSec, pad = True, fill_value = 0)
        stream_aligned[iterStation] = traceShifted
        

        tLagsInSec.append(tLagInSec)
        normFacs.append(maxiTwin)



        if plotShifting:
            plt.figure(figsize=(10,7))
            plt.plot(trace.times(), wavelet/max(wavelet), c = 'k', lw = 2, label = 'wavelet')
            plt.axvline(tt_DP_stat, -1, 1, c = 'k', lw = 2)
        # plt.plot(trace.times(), trace.data/max(abs(trace.data)), label = 'data')
            plt.plot(trace.times(), traceXcorr.data/max(abs(traceXcorr.data)), c= 'r', ls = '--', lw = 2, label = 'original')
            plt.plot(traceShifted.times(), traceShifted.data/max(abs(traceShifted.data)), c = 'b', ls = '-', lw = 2, label = 'shifted')
            plt.legend(loc = 1)
            plt.xlim(0,2)
            plt.title('Alignment Station' + station)
        
    return stream_aligned


def apply_maxAmp(stream, maxAmpWin, tt_DP, method = 'max'):
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
   
   # for tr in stream_mod: 
        trace.stats.starttime = sTime
    
    return stream_mod


def get_traveltimes(index, stations):

    path_file = './META/PStraveltimes' 
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

def preprocess_data(stream, iterQuake, freqRange,alignTo1D = True, maskDirectWaves = True, normalize_stream = True):

    ### quick check of nans in trace
    for tr in stream: 
        if np.isnan(tr.data).any():
            # Replace NaNs with 0
            tr.data = np.nan_to_num(tr.data, nan=0.0)

    # tapering
    stream.taper(0.05, side = 'both')
    stream.detrend('linear').detrend('constant')


    freqmin = freqRange[0]
    freqmax = freqRange[1]

    if not (freqmin == None) or (freqmax == None):
        stream.filter('bandpass', freqmin = freqmin, freqmax = freqmax, zerophase = True, corners = 1)
    
    stations = [tr.stats.station for tr in stream]
    tt_DP, tt_DS = get_traveltimes(iterQuake, stations)
    

    corrFacs = tt_DP - 0.5
    tt_DP -= corrFacs
    tt_DS -= corrFacs

    # if alignTo1D  == True:
    #     trimLenforXcorr = 0.04
    #     maxAmpWin = 0.08 #0.08
    #  #   stream = apply_xcorr_wavelet(stream, tt_DP, trimLenforXcorr = trimLenforXcorr)
    #     stream = apply_maxAmp(stream, maxAmpWin, tt_DP)

    if maskDirectWaves:
        for tr, iterTr in zip(stream, range(len(stream))):
            ### mute P and S waves
            sampRate = tr.stats.sampling_rate
            tt_DP_samp = np.round(tt_DP[iterTr]*sampRate)
            tt_DS_samp = np.round(tt_DS[iterTr]*sampRate)
            Pwin_delete_samp = 6

            tr.data[0:int(tt_DP_samp+Pwin_delete_samp)] = 0
            tr.data[int(tt_DS_samp)-6:] = 0

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
        evString = getEventString(df, iterQuake)
        evString = evString.split('.mseed')[0]
        fNameCombisEvent = pathCombis + evString + '.csv'
    
        binCombisEvent = pd.read_csv(fNameCombisEvent)
        return binCombisEvent
