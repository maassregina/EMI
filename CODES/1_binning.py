# %% -----------------------------------------------------
##########################################################
##########################################################
##########################################################
##########################################################

import os
import numpy as np 
import pandas as pd
import multiprocessing
import time
import functions_migration as fm


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rc('font', size=25)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 



# %% ------------------------------------------ MAIN FUNCTION
##########################################################
##########################################################
##########################################################
##########################################################


#-- main function
def get_binSolutions(iterQuake, saveCombinations = True):

    evString = fm.getEventString(dfEvents, iterQuake, config=None).split('.mseed')[0]
    pathSave_file = path_results_tmp + evString

    evLon = evLons[iterQuake]
    evLat = evLats[iterQuake]
    evZ = evDepths[iterQuake]

    path_file = path_traveltimes + 'results_' + wavetype +'_eventNr' + str(iterQuake)


    try:
        dataFile=pd.read_csv(path_file+".csv")
    except:
        return None

    reflectorDepths = np.array(dataFile['Reflector_depth'])


    #--------- NOW LOOP OVER GRID CELLS/BINS 
    dfForAllBins = []

    binNr = 0
    for i in range(rotated_xx.shape[0] - 1):

        #### parameters of grid cell. DEFINE CELL. 
        for j in range(rotated_yy.shape[1] - 1):

            cell_vertices = [(rotated_xx[i, j], rotated_yy[i, j]),
                         (rotated_xx[i+1, j], rotated_yy[i+1, j]),
                         (rotated_xx[i+1, j+1], rotated_yy[i+1, j+1]),
                         (rotated_xx[i, j+1], rotated_yy[i, j+1])]
    
            binCombis = [] ## for given cell.

            results = []

            for station in stations:
                stationString = station
                if station.startswith('ARR'):
                    stationString = station[0:3] + '0' + station[3:]
                staStringX = stationString + '_X_' + wavetype + '[km]'
                staStringY = stationString + '_Y_' + wavetype + '[km]'
                result = fm.extract_values_within_grid(dataFile, cell_vertices, staStringX, staStringY)
            
                result = fm.keep_columns_by_name(result, stationString)
                results.append(result)

            dfForBin = fm.combine_dataframes(results, 'Reflector_depth')
            
            Nrows = dfForBin.shape[0]
            vector = np.ones(Nrows)*binNr
            vector = vector.astype(int)
            dfForBin['BinNr'] = vector

            col_to_move = 'BinNr'
            column = dfForBin.pop(col_to_move)
            dfForBin.insert(0, col_to_move, column)
            dfForBin['BinNr'].astype(int)

            ### assign bin number for this big df
            ### combine these results to get one pandas dataframe per bin

            dfForAllBins.append(dfForBin)
            binNr += 1
    

    ######### concatenate to big matrix that contains all combinations for a given bin Number and reflectordepths
    ### stack vertically to create ONE df for each earthquakes
    dfAllBinsForEvent = dfForAllBins[0]
    for df in dfForAllBins[1:]:
            dfAllBinsForEvent = pd.concat([dfAllBinsForEvent, df], axis=0, ignore_index=True)

    if saveCombinations == True:
        print('--- Save combinations for Event No. ' + str(iterQuake))
        dfAllBinsForEvent.to_csv(pathSave_file + '.csv', index=False)





# %% ------------------------------------------ PATHS
##########################################################

path_meta = '../META/'
path_data = '../DATA/'
path_results_root = '../RESULTS/'
path_traveltimes = '../META/Traveltime_matrices/'


### reference coordinates in deg
lonRef = -16.8834
latRef = 65.6785

### longitude and latitude of IDDP-1
lonIDDP, latIDDP  = -16.76413050, 65.71588535



# %% ------------------------------------------ USER-DEFINED PARAMETERS
##########################################################
##########################################################
##########################################################
##########################################################

#-------------------------------- GENERAL PARAMETERS
#### define phase to be analysed.
wavetype = "RP" #'RP' corresponds to primary PP reflection

#### path for saving depending on the wavetype
path_results = path_results_root + 'Wavetype_' + str(wavetype) + '/'

#### force analysis even if datafiles already exists? (Default is False)
forceOverwriteData = True

#### plot map showing the study area and grid?
plotMap = True

#### number of cores
Ncores = 48

#-------------------------------- SELECT EARTHQUAKES TO BE INCLUDED IN THE ANALYSIS
dfEvents = pd.read_csv(path_meta + 'earthquake_info.csv')
indices = np.array(dfEvents.index)

evLons = dfEvents['Longitude'].values
evLats = dfEvents['Latitude'].values
evDepths = dfEvents['Depth'].values

#-------------------------------- SELECT STATIONS TO BE INCLUDED IN THE ANALYSIS

dfStats = pd.read_csv(path_meta + 'station_info.csv')
stations = dfStats['STATION'].values
stLons = dfStats['LONGITUDE'].values
stLats = dfStats['LATITUDE'].values

##### transform values into right coordinate system.
stX,stY = fm.LonLatToKm(stLons, stLats, coordsRefLonLat = [lonRef, latRef])
evX, evY = fm.LonLatToKm(evLons, evLats, coordsRefLonLat = [lonRef, latRef])


#-------------------------------- GRIDDING OF AREA
### define the model space (cartesian coordinates): [[x_min, x_max], [y_min, y_max]]
modelSpaceLimits = [[4.5,7.2], [2.6, 5]]

### grid increment - can be several (e.g., [0.1, 0.07]), code will loop over grid increments
gridIncX_list = [0.2] #spacing in x direction, dx
gridIncY_list = [0.2] #spacing in y direction, dy

### rotation angle of the grid  - can be several (e.g., [-7, 40]), code will loop over grid increments)
angle_list = [-7]


# %% ------------------------------------------ PROGRAM START
##########################################################
##########################################################
##########################################################
##########################################################


######################## LOOP OVER GRID INCREMENTS AND ANGLES
for angle in angle_list:
    for gridIncX, gridIncY in zip(gridIncX_list, gridIncY_list):

        ### grid increment
        gridInc = [gridIncX, gridIncY]
        print('gridIncs ' + str(gridIncX) + ', ' + str(gridIncY) + '\nangle' + str(float(angle)))


        #-------------------------------- REDEFINE PATH SAVE ROOT BASED ON CHOSEN GRID INCREMENTS
        path_results_tmp = path_results + 'GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle) + '/'
        path_results_tmp += 'BINNING/'

        if os.path.exists(path_results_tmp)==False:
                os.makedirs(path_results_tmp)


        #-------------------------------- CREATE GRID

        gridLengths = [gridInc[0], gridInc[1]] #in x and y direction, in km
        xRef, yRef = fm.LonLatToKm([lonIDDP], [latIDDP], coordsRefLonLat = [lonRef, latRef])
        rotated_xx, rotated_yy = fm.create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(xRef, yRef))


        NlinesY = rotated_xx.shape[0]
        NlinesX = rotated_xx.shape[1]


        center_points = [] 
        
        for i in range(NlinesY - 1):
                for j in range(NlinesX  - 1):
                    center_point = fm.compute_center_point(rotated_xx[i, j],rotated_yy[i, j], rotated_xx[i+1, j], rotated_yy[i+1, j], rotated_xx[i+1, j+1], rotated_yy[i+1, j+1], rotated_xx[i, j+1], rotated_yy[i, j+1])                                 
                    center_points.append(center_point)
        Nbins = len(center_points)

        #------------------------------------------ PLOTTING OF GRID 

        ####### Boundaries of cells are plotted with blue color
        ####### Numbers are plotted at midpoints of cells
        ####### Map will be saved in path_results_tmp folder

        ### Plotting
        if plotMap: 
            fig = plt.figure(figsize=(60,30))
            mapname = 'GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle)

            matplotlib.rc('font', size=60)
            matplotlib.rc('xtick', labelsize=50) 
            matplotlib.rc('ytick', labelsize=50) 

            gs = gridspec.GridSpec(4, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0:4, 0])
            ax1.set_ylabel("Y [km]", fontsize = 40)
            ax1.set_xlabel("X [km]", fontsize = 40)
            ax1.plot(stX, stY, "ko", alpha = 1, markersize = 20)
            ax1.plot(xRef, yRef, 'ko', markersize = 50)

            ax1.plot(evX, evY, 'r*', markersize = 30)
            ax1.plot(rotated_xx, rotated_yy, 'bo', markersize=20)
            plt.gca().set_aspect('equal', adjustable='box')
            ax1.grid(True)
            ax1.set_xlabel('X [km]', fontsize = 80)
            ax1.set_ylabel('Y [km]', fontsize = 80)

            ax2 = fig.add_subplot(gs[0:4, 1])
            ax2.plot(rotated_xx, rotated_yy, 'bo', markersize=10)
            ax2.set_ylabel("Y [km]", fontsize = 80)
            ax2.set_xlabel("X [km]", fontsize = 80)
            ax2.plot(stX, stY, "ko", alpha = 1, markersize = 20)
            ax2.plot(evX, evY, 'r*', markersize = 30, zorder = -10)
            ax2.plot(xRef, yRef, 'ko', markersize = 50)

            for iterBin in range(Nbins):
                ax2.text(center_points[iterBin][0], center_points[iterBin][1], str(iterBin), fontsize = 40, zorder = 10, horizontalalignment = 'center', verticalalignment = 'center')

            plt.suptitle(mapname)
            plt.tight_layout()

            ### Save fig
            plt.savefig(path_results_tmp + 'MAP_' + mapname + '.png', bbox_inches = 'tight')



        # ------------------------------------------ ACTUAL ANALYSIS
        ##########################################################
        ##########################################################
        ##########################################################
        ##########################################################

        start = time.time()

        #-------------------------------- 1. CHECK IF RESULTS ALREADY EXIST FOR GIVEN EVENT
        if forceOverwriteData == True:
            indices_EventsToProcess = indices

        if forceOverwriteData == False:
            indices_EventsToProcess = []

            for iterQuake in indices:
                    evString = fm.getEventString(dfEvents, iterQuake, config=None).split('.mseed')[0]
                    fNameCombisEvent = path_results_tmp + evString + '.csv'

                    if os.path.exists(fNameCombisEvent)==False: 
                        indices_EventsToProcess.append(iterQuake)

            if (len(indices_EventsToProcess) == 0):
                print(' --------- ALL FILES ALREADY EXIST, proceed with other parameters -----------')
                continue


       #-------------------------------- 2. RUN FUNCTION FOR EVENTS FOR WHICH RESULTS DONT YET EXIST

        if len(indices_EventsToProcess) > 0:
            print(str(len(indices_EventsToProcess)) +' Events must be processed - run code in parallel, numer of cores = ' + str(Ncores))
            def parallel_multi(inputs):
                pool = multiprocessing.Pool(Ncores)  
                results = pool.map(get_binSolutions, indices_EventsToProcess) 
                pool.close()  
                pool.join()  
                return results

        if __name__ == "__main__":
            streams_ori = parallel_multi(indices_EventsToProcess)



        end = time.time()


        ##########################################################
        ##########################################################
        ##########################################################
        ##########################################################

        print('------------ BINNING COMPLETED: all files saved. ----------------')
        print(str((end - start)/60) + ' MINUTES NEEDED TO PROCESS EVENTS' )

# %% -------------------------------------------------------------------------------------------------