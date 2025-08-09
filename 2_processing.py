# %% ------------------------------------------
##########################################################
##########################################################
##########################################################
##########################################################

import os
import sys
import copy

import functions_migration as fm

import numpy as np 
import pandas as pd
import math
import scipy

import obspy 
from obspy import read, Stream

import concurrent.futures
import time


# %% ------------------------------------------
##########################################################
##########################################################
##########################################################
##########################################################


def get_combis_for_event_csv(iterQuake, dfEvents):

    reflectorDepthsAll = fm.returnReflectorDepths(wavetype)
    reflectorDepthsAll = [float("{:.2f}".format(num)) for num in reflectorDepthsAll]
    NdepthsAll = len(reflectorDepthsAll)

    ## IDDP depth for correct assigning of depths
    Z_IDDP = -542.41/1000  ##not 616.712

    ### load binned combinations
    evString = fm.getEventString(dfEvents, iterQuake)
    evString = evString.split('.mseed')[0]
    fNameCombisEvent = path_binning + evString + '.csv'
    df_iterQuake= pd.read_csv(fNameCombisEvent)

    stream = streamsFilt[iterQuake]
    sampRate = stream[0].stats.sampling_rate

    stNames = [tr.stats.station for tr in stream]

    bins_depths = []
    bins_ampl= []

    binNrsWithCombis = np.array(df_iterQuake['BinNr'])
    reflecDepths = np.array(df_iterQuake['Reflector_depth'])

    ### allocate matrix with zeros
    matrix = np.zeros((NdepthsAll , NbinsAll))
    matrixAmpl = [[[] for _ in range(NbinsAll)] for _ in range(NdepthsAll)]

    division_matrix = np.zeros((NdepthsAll , NbinsAll)) ## in order to calculate average later 


    ### added on 15-01-25
    # Initialize the matrix with placeholder values
    CDPgatherInfo = [[[] for _ in range(NbinsAll)] for _ in range(NdepthsAll)]
    CDPgather_streams  = [[Stream() for _ in range(NbinsAll)] for _ in range(NdepthsAll)]

    ######## extract direct wave arrivals - in order to later avoid them.
    index = indices[iterQuake]
    tt_DP, tt_DS  = fm.get_traveltimes(index, stations)

    corrFacs = tt_DP - 0.5
    tt_DP -= corrFacs
    tt_DS -= corrFacs

    for iterCount, station in zip(range(len(stations)), stations):
        try:
            trace = stream.select(station = station)[0]
        except: ### station does not exist
            continue

        if np.all(trace.data == 0) == True:  ###added on 08-04-25, denn dann sollte das nicht sum fold geadded werden etc. 
           #     iterCount += 1
                continue
        stationString = station
        if station.startswith('ARR'):
            stationString = station[0:3] + '0' + station[3:]

        try:
            tt_vec = df_iterQuake[stationString + '_tt_' + wavetype + '[s]']
            tt_vec = np.array(tt_vec)
        except:
            continue

        corrFac = corrFacs[iterCount] #### NEW

        idx_nonzero = np.where(tt_vec != 0)[0]


        ### loop through corresponding rows in df_iterQuake
        for iterRow in idx_nonzero:
            tt = tt_vec[iterRow] - corrFac
            
            ### extract amplitude
            Npick = int(round(tt*sampRate,0)) ###okay with int and round?? not precise enough?
            amp = trace.data[Npick]

            ## now assign amp to correct binNr in matrix and reflector depth
            idxBin = int(binNrsWithCombis[iterRow])
            refDepth = reflecDepths[iterRow]
            refDepth -= Z_IDDP #IDDP depth is added to get depth below IDDP and not depth b.s.l. 
            refDepth = float("{:.2f}".format(refDepth))
            idxDepth = reflectorDepthsAll.index(refDepth)

            matrix[idxDepth, idxBin] += amp
            division_matrix[idxDepth, idxBin] += 1
            matrixAmpl[idxDepth][idxBin].append(amp)

            CDPgatherInfo[idxDepth][idxBin] += [station, iterQuake, tt]
            CDPgather_streams[idxDepth][idxBin] += trace

    fold = division_matrix

    return [matrix, fold, CDPgatherInfo, CDPgather_streams]


# %%########################################################################################
############################################################################################
####-------------------------------- USER-DEFINED PARAMETERS
############################################################################################
############################################################################################
############################################################################################

#-------------------------------- GENERAL PARAMETERS

#### define phase to be analysed.
wavetype = 'RP'

#### print statements? 
verbose = True

#### plotMap of station-earthquake geometry?
plotMap = True

#### define component to be analysed (default is 'Z).
# RP: P-P reflection, GP: ghost P (P-P-P) reflection, RSP: S-P conversion, RS: S-S reflection, GS: Ghost S (S-S-S) reflection

if wavetype in ['RP', 'GP', 'RSP']: 
    component = 'Z'
elif wavetype in ['RS', 'GS']:
    component = 'R'

### reference coordinates in deg/ reference coordinate system used here. 
lonRef = -16.8834
latRef = 65.6785

### longitude and latitude of IDDP-1
lonIDDP, latIDDP  = -16.76413050, 65.71588535


#-------------------------------- PATHS
path_earthquakes = '/DATA/'
path_binning_root  = './RESULTS/Wavetype_' + wavetype 
path_data = './DATA/'
path_results_root = './RESULTS/Wavetype_' + wavetype
path_meta = './META/'



#-------------------------------- EARTHQUAKES, STATIONS AND GRID
dfEvents = pd.read_csv(path_meta + 'earthquake_info.csv')
dfStats = pd.read_csv(path_meta + 'station_info.csv')

indices = np.array(dfEvents.index)

### define the model space (cartesian coordinates): [[x_min, x_max], [y_min, y_max]]
modelSpaceLimits = [[4.5,7.2], [2.6, 5]]

### grid increment - can be several (e.g., [0.1, 0.07]), code will loop over grid increments
gridIncX_list = [0.2] #spacing in x direction, dx
gridIncY_list = [0.2] #spacing in y direction, dy

### rotation angle of the grid  - can be several (e.g., [-7, 40]), code will loop over grid increments)
angle_list = [-7]


#-------------------------------- PROCESSING PARAMETERS
### Normalization?
normalize_stream = True #if True, traces will be normalized by their RMS

### Muting of the direct waves?
maskDirectWaves = True # will mask direct P and S waves as well as S coda

### aligning to 1D model?
alignTo1D = False #not needed here as data are already preprocessed such that P wave is aligned

### freqquency range, can bin list. [[f1, f2], [f3,4], ...]
freqRanges = [[5,25]]

### save matrix for each event? if False, only stacked results will be saved
saveIndividualMatrices = True

#####################################################
print('___________________________________________________________________________________')
print('--> Data will be taken from: ' + path_data)
print('--> ' + str(len(indices)) + ' Events will be processed.')
print('--> Analysis is carried out for wavetype: ' + wavetype)


if plotMap: 
    import matplotlib.pyplot as plt

    ### extract values for plotting
    indices = np.array(dfEvents.index)
    evLons = dfEvents['Longitude'].values
    evLats = dfEvents['Latitude'].values
    stations = dfStats['STATION'].values
    stLons = dfStats['LONGITUDE'].values
    stLats = dfStats['LATITUDE'].values

    ##### transform values into right coordinate system.
    stX,stY = fm.LonLatToKm(stLons, stLats, coordsRefLonLat = [lonRef, latRef])
    evX, evY = fm.LonLatToKm(evLons, evLats, coordsRefLonLat = [lonRef, latRef])

    plt.figure(figsize = (15,10))
    plt.scatter(evLons, evLats, color = "r", s = 80, alpha =1, zorder = 1)
    plt.scatter(stLons, stLats, color = "k", s = 80, alpha =1, zorder = 1)
    plt.title('Station-earthquake configuration ')
    plt.grid()
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')



# %%########################################################################################
############################################################################################
####-------------------------------- LOAD DATA
############################################################################################
############################################################################################
############################################################################################

START = time.time()

print('___________________________________________________________________________________')
print('.... load real data')


streams_ori = []
for iterQuake in range(dfEvents.shape[0]):
    stream = Stream()
    for config in ['L1', 'L2', 'ARR']:
        evString = fm.getEventString(dfEvents, iterQuake, config=config)
        stream_config = read(path_data + evString)
        stream += stream_config
    stream.trim(stream[0].stats.starttime + 5, stream[0].stats.starttime + 10, pad= True, fill_value = 0)
    
    # remove stations that are not in station list
    for tr in stream:
        if tr.stats.station not in stations:
            stream.remove(tr)

    streams_ori.append(stream)


# %% ------------ remove response 

def remove_response(stream,path_inv=None, pre_filt = None, output = "DISP"):
    if path_inv == None:
        path_inv="/home/maass/Desktop/PhD/Script_SPseis/Inventory/KF_inventory.xml"
    stream.remove_response(inventory=path_inv, pre_filt=pre_filt, output=output, taper = True, zero_mean=True,
                   water_level=60, plot=False) 
  #  inv = obspy.read_inventory(path_inv)
  #  stream.remove_sensitivity(inv)  
    
    return stream

def remove_responses_from_streams(iterQuake):
    
    stream = streams_ori[iterQuake]
    if np.all(np.array(stream) == 0) == True:
        return stream
    
    print(iterQuake, ' remove instrument response')
    # try:
    pre_filt = [2, 5, 25, 30]
    #     stream = remove_response(stream, path_inv=None, pre_filt = pre_filt, output = "DISP") ##remove response
    # except:
    for tr in stream:
        if np.all((tr.data == 0)) == False:
            tr = remove_response(tr, path_inv=None, pre_filt = pre_filt, output = "VEL")
    return stream

from joblib import Parallel, delayed


num_cores = 48
streams_ori = Parallel(n_jobs=num_cores, verbose=0)(delayed(remove_responses_from_streams)(iterQuake) for iterQuake in range(len(streams_ori)))

# %%
for iterQuake in range(dfEvents.shape[0]):
    stream = streams_ori[iterQuake]
    for config in ['L1', 'L2', 'ARR']:
        stream_tmp = stream.select(station = config + '*')
        evString = fm.getEventString(dfEvents, iterQuake, config=config)
        fName_src= path_data + evString
        stream_tmp.write(fName_src, format="MSEED") 


# %%########################################################################################
############################################################################################
####-------------------------------- START PROGRAM: LOOP OVER ANGLES AND GRID INCREMENTS 
############################################################################################
############################################################################################
############################################################################################


for angle in angle_list:
    for gridIncX, gridIncY in zip(gridIncX_list, gridIncY_list):

        print('___________________________________________________________________________________')
        print('ANALYSIS FOR gridIncs ' + str(gridIncX) + ', ' + str(gridIncY) + ', angle = ' + str(float(angle)))

        path_binning = path_binning_root + '/GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle)+'/' 
        path_binning +=  'BINNING/'

        path_results = path_results_root + '/GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle)+'/' 
        path_results +=  'NUMPY_MATRICES/'
        if os.path.exists(path_results )==False:
                os.makedirs(path_results )
    

        if os.path.isdir(path_binning) == False:
            print('THESE GRIDDING PARAMETERS DO NOT EXIST')
            break

        # ------------------------ REFLECTOR DEPTHS AND GRIDDING

        ### GRIDDING
        gridLengths = [gridIncX, gridIncY]

        xRef, yRef = fm.LonLatToKm([lonIDDP], [latIDDP], coordsRefLonLat = [lonRef, latRef])
        rotated_xx, rotated_yy = fm.create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(xRef, yRef))

        #rotated_xx, rotated_yy = create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(0,0))

        ######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
        Nlines_x = rotated_xx.shape[1] - 1
        Nlines_y = rotated_xx.shape[0] -1
        NbinsAll = (Nlines_x * Nlines_y)

            
        # ------------------------ PREPROCESSING

        for freqmin, freqmax in freqRanges:
            freqRange = [freqmin, freqmax]

            print('FREQUENCY RANGE ' + str(freqmin) + '-' + str(freqmax) + ' HZ')


            print('.... pre-processing')


            ### copy original data to allow for multiple preprocessing workflows
            streams = copy.deepcopy(streams_ori)

            streamsFilt = []
            for iterQuake in range(len(streams)):
                stream = streams[iterQuake]
                streamProc = fm.preprocess_data(stream, iterQuake, freqRange, alignTo1D = alignTo1D, maskDirectWaves = maskDirectWaves, normalize_stream = normalize_stream)
                streamsFilt.append(streamProc)

            sampRate = streamsFilt[0][0].stats.sampling_rate 

            # ------------------------ STACK AMPLITUDES FROM SAME QUAKE IF THEY FALL INTO SAME DEPTH IN GIVEN BIN
            print('.... stack amplitudes in each bin')
            inputs=list(np.arange(0,len(streamsFilt),1))

            Ncores = 48
            with concurrent.futures.ThreadPoolExecutor(Ncores) as executor:
                futures  = [executor.submit(get_combis_for_event_csv, iterQuake, dfEvents) for iterQuake in inputs]



            # ------------ STACKING OVER ALL EVENTS

            print('.... stacking over events')

            NdepthsAll = futures[0].result()[0].shape[0]

            matrix = np.zeros((NdepthsAll, NbinsAll))
            fold = np.zeros((NdepthsAll, NbinsAll))

            ##################
            matricesIndividual = []
            for iterEv in range(len(inputs)):
                try:
                    m_ = futures[iterEv].result()[0] ##matrix corresponding to event
                    matrix += m_    ## matrix divided by fold

                    f_ = futures[iterEv].result()[1] ## fold corresponding to event
                    fold += futures[iterEv].result()[1]
                    
                    f_[f_ == 0] = 1 ## avoiding division through zero
                    matricesIndividual.append(m_/f_)

                except: 
                    print('ERROR FOR EVENT NO ' + str(iterEv) + ' --> CREATE TRAVELTIME FILE WITH MATLAB FOR THIS EVENT IN FUTURE')
                    continue

            fold[fold == 0] = 1
            matrix = matrix/fold 

            # ------------------------- SAVING 


            postproc_string = 'freqRange' + str(freqmin) + '-' + str(freqmax)

            print('.... saving of 3D numpy matrix in ' + path_results + postproc_string)


            ################ DATA MUST BE RE-ORGANISED INTO X- AND Y- DIRECTIONS
            ### THUS LOCATIONS OF BINS MUST BE KNOWN IN SPACE 

            matrix3D = matrix.reshape(NdepthsAll, Nlines_y, Nlines_x)
            fold3D = fold.reshape(NdepthsAll, Nlines_y, Nlines_x)

            ### SAVING 
            np.save(path_results + '3Dmatrix_' + postproc_string  + '.npy', matrix3D)
            np.save(path_results + 'Fold_' + postproc_string  + '.npy', fold3D)

            if saveIndividualMatrices:
                path_resultsIndividual = path_results + 'Individual/'
                if os.path.exists(path_resultsIndividual)==False:
                    os.mkdir(path_resultsIndividual)

                for iterEvent in range(len(matricesIndividual)):
                    fileName = 'EventIndex_' + str(np.array(dfEvents.index)[iterEvent]) + '_' + postproc_string
                    
                    m_ = matricesIndividual[iterEvent]
                    m_ = m_.reshape(NdepthsAll, Nlines_y, Nlines_x)

                    np.save(path_resultsIndividual + fileName + '.npy', m_)

END = time.time()
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')
print('----------------------------------- END PROGRAM -----------------------------------')
print('---------------' + str((END - START)/60) + ' MINUTES NEEDED IN TOTAL --------------')
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')

# %%
