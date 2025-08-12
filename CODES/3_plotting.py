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
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rc('font', size=25)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 




def plot_grid(iterLine, imageAlong, ax1, individualEvent = None, fold = [], depthIdxFold = 0, cmap = 'Greys', plotEarthquakes = True, coordsTransform = [5.47956943, 4.16858575], angle = -7):
    
    ### Plotting of map
    mapname = 'GridIncr' + str(gridIncX) + '-' + str(gridIncY)

    transX = coordsTransform[0]
    transY = coordsTransform[1]


    ax1.plot(stX - transX, stY - transY,'ko', alpha = 0.6, markersize = 60)
    ax1.plot(xIDDP[0]- transX, yIDDP[0] - transY,'yo', alpha = 1, markersize = 140)

    if (individualEvent == None):
            alpha = 1
            if len(fold) > 0:
                alpha = 0.5
            if plotEarthquakes == True:
                ax1.plot(evX - transX, evY - transY, 'r*', zorder = 18, markersize = 150, alpha = alpha, markeredgecolor = 'black', markeredgewidth = '10')
    elif individualEvent: 
        ax1.plot(evX[individualEvent] - transX, evY[individualEvent] - transY, 'r*', markersize = 125, alpha = 1)

    ax1.plot(rotated_xx - transX, rotated_yy - transY, 'bo', markersize=40, alpha = 0.3, zorder =-1)
    
    if len(fold) > 0:
        vmin = thresholdFold
        vmax = 15
        fold[fold < vmin] = np.nan
        im = ax1.pcolormesh(rotated_xx - transX, rotated_yy - transY, fold[depthIdxFold,:,:], cmap = cmap, vmin = vmin, vmax = vmax, alpha = 0.8)

    ax1.grid(False)
    ax1.set_ylabel("Y [km]")
    ax1.set_xlabel("X [km]")

    mapname = 'GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle)
    ax1.set_title(mapname)

    if imageAlong == 'Image_alongY':
        xmin = rotated_xx_angle0[0][iterLine]
        xmax = rotated_xx_angle0[0][iterLine + 1]
        ymin = np.min(rotated_yy_angle0)
        ymax = np.max(rotated_yy_angle0)

        ### coordinates of the rectangle
        x_rect = [xmin, xmax, xmax, xmin]
        y_rect = [ymin, ymin, ymax, ymax]

        x_rect_rot, y_rect_rot = fm.rotate_point((x_rect, y_rect), angle, refCoords = (xRef, yRef))
        ax1.fill(x_rect_rot - transX, y_rect_rot - transY, color='b', alpha=0.4, label='Shaded Area')  # Fill area


    if imageAlong == 'Image_alongX':
        ymin = rotated_yy_angle0[iterLine][0]
        ymax = rotated_yy_angle0[iterLine+1][0]
        xmin = np.min(rotated_xx_angle0)
        xmax = np.max(rotated_xx_angle0)

        x_rect = [xmin, xmax, xmax, xmin]
        y_rect = [ymin, ymin, ymax, ymax]

        x_rect_rot, y_rect_rot = fm.rotate_point((x_rect, y_rect), angle, refCoords = (xRef, yRef))
        ax1.fill(x_rect_rot - transX, y_rect_rot - transY, color='b', alpha=0.4,edgecolor = 'k', label='Shaded Area')  # Fill area
        

    return ax1


def plot_vertical_cross_section(matrix, iterLine, freqRange, scaFac = 1, figsize = None, yLim = None, xLim = None, individualEvent = None, perc_ampl = 97, fold = [], depthIdxFold = 0, imageAlong = 'Image_alongY'):
    
    
    matplotlib.rc('font', size=150)
    matplotlib.rc('xtick', labelsize=120) 
    matplotlib.rc('ytick', labelsize=120) 

    freqmin = freqRange[0]
    freqmax = freqRange[1]

    Nlines = matrix.shape[1]
    if figsize == None:
        figsize = (40,15)
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(4, 10, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    ax = fig.add_subplot(gs[0:4, 4:7])
    ax3 = fig.add_subplot(gs[0:4, 7:10], sharey = ax)

    ### create x axes
    if imageAlong == 'Image_alongX':
        xLim_list = center_points_XX[iterLine,:] - transX

    if imageAlong == 'Image_alongY':
        xLim_list = center_points_YY[:, iterLine] - transY


    ax1 = plot_grid(iterLine, imageAlong, ax1, individualEvent = individualEvent, fold = fold, depthIdxFold=depthIdxFold)

    ax1.set_xlim(map_limits[0])
    ax1.set_ylim(map_limits[1])
   

    for iterBin in range(Nlines):
        count = xLim_list[iterBin]
        amplitudes = matrix[:, iterBin]

        if len(amplitudes) == 0: 
            ax.plot(count, np.nan, 'X', lw = 2)

        else:
            amplitudes = amplitudes/scaFac
            amplitudes += count

            ax.plot(amplitudes, reflectorDepthsAll, '-', c = 'k')
            if colorMode == 'k':
                ax.fill_betweenx(reflectorDepthsAll,count,amplitudes,where=(amplitudes>count),color='k')

            elif colorMode == 'r':
                ax.fill_betweenx(reflectorDepthsAll,count,amplitudes,where=(amplitudes>count),color='r')
                ax.fill_betweenx(reflectorDepthsAll,count,amplitudes,where=(amplitudes<count),color='b')
        
    YTICKS = np.arange(min(reflectorDepthsAll), max(reflectorDepthsAll), 0.5)

    ax.set_yticks(YTICKS)        

    ax.set_title('Fequency range ' + str(freqmin) + '-' + str(freqmax) + ' Hz')

    ###
    X = np.arange(0, matrix.shape[1], 1)
    Y = reflectorDepthsAll
    
    cmap = 'Greys'

    matrix[np.isnan(matrix)] = 0
    matrix[~np.isfinite(matrix)] = 0
    vm=np.percentile(np.absolute(matrix), perc_ampl)

    im = ax3.pcolormesh(xLim_list, Y, matrix, cmap = cmap, vmin = -vm, vmax =vm)

    ax.set_ylabel('Depth below IDDP1 [km]')
    ax3.set_ylabel('Depth below IDDP1 [km]')
    if imageAlong == 'Image_alongY':
        ax.set_xlabel('Y [km]\n S --> N')
        ax3.set_xlabel('Y [km]\n S --> N')
    if imageAlong == 'Image_alongX':
        ax.set_xlabel('X [km]\n E --> W')
        ax3.set_xlabel('X [km]\n E --> W')
    if yLim == None:
        ax.set_ylim(6,1)
        ax3.set_ylim(6,1)
    else: 
        ax.set_ylim(yLim[0],yLim[1])
        ax3.set_ylim(yLim[0],yLim[1])
    if xLim: 
        ax.set_xlim(xLim[0], xLim[1])
        ax3.set_xlim(xLim[0], xLim[1])

    plt.suptitle(imageAlong + ', line no.' + str(iterLine) + ' - Stacked Earthquakes')


    return fig


# %%########################################################################################
############################################################################################
####-------------------------------- DEFINE CRUCIAL VARIABLES
############################################################################################
############################################################################################
############################################################################################
#### define phase to be analysed.
wavetype = 'RP' #RP is for primary P-P reflection (see also 2_processing.py)

### reference coordinates in deg/ reference coordinate system used here. 
lonRef = -16.8834
latRef = 65.6785

### longitude and latitude of IDDP-1
lonIDDP, latIDDP  = -16.76413050, 65.71588535
xIDDP, yIDDP = fm.LonLatToKm([lonIDDP], [latIDDP], coordsRefLonLat = [lonRef, latRef])


#-------------------------------- PATHS
path_earthquakes = '/DATA/'
path_binning_root  = './RESULTS/Wavetype_' + wavetype 
path_data = './DATA/'
path_results_root = './RESULTS/Wavetype_' + wavetype
path_meta = './META/'

#------------------------------- EARTHQUAKES, STATIONS AND GRID
dfEvents = pd.read_csv(path_meta + 'earthquake_info.csv')
dfStats = pd.read_csv(path_meta + 'station_info.csv')

indices = np.array(dfEvents.index)
evLons = dfEvents['Longitude'].values
evLats = dfEvents['Latitude'].values
stations = dfStats['STATION'].values
stLons = dfStats['LONGITUDE'].values
stLats = dfStats['LATITUDE'].values

##### transform values into right coordinate system.
stX,stY = fm.LonLatToKm(stLons, stLats, coordsRefLonLat = [lonRef, latRef])
evX, evY = fm.LonLatToKm(evLons, evLats, coordsRefLonLat = [lonRef, latRef])


### define the model space (cartesian coordinates): [[x_min, x_max], [y_min, y_max]]
modelSpaceLimits = [[4.5,7.2], [2.6, 5]]

### grid increment and angle
gridIncX = 0.07
gridIncY = 0.07

### rotation angle of the grid 
angle = -7

reflectorDepthsAll = fm.returnReflectorDepths(wavetype)
reflectorDepthsAll = [float("{:.2f}".format(num)) for num in reflectorDepthsAll]


# %% ------ GRIDDING

### Create rotated grid.
gridLengths = [gridIncX, gridIncY]
xRef, yRef = fm.LonLatToKm([lonIDDP], [latIDDP], coordsRefLonLat = [lonRef, latRef])
map_limits = [[-0.8, 0.8], [-0.8, 0.75]]
rotated_xx, rotated_yy = fm.create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(xRef, yRef))
rotated_xx_angle0, rotated_yy_angle0 = fm.create_rotated_grid(modelSpaceLimits, gridLengths, 0, refCoords =(xRef, yRef))

### extract boundaries 
x0 = rotated_xx[0][0]
x1 = rotated_xx[0][-1]
y0 = rotated_yy[0][0]
y1 = rotated_yy[0][-1]        

###
Nlines_x = rotated_xx.shape[1] - 1
Nlines_y = rotated_xx.shape[0] -1
NbinsAll = (Nlines_x * Nlines_y)
offsets_x = np.linspace(x0, x1,Nlines_x)


center_points = [] 
center_points_XX = np.zeros((Nlines_y, Nlines_x))
center_points_YY = np.zeros((Nlines_y, Nlines_x))

###
for i in range(Nlines_y):
        for j in range(Nlines_x):
            center_point = fm.compute_center_point(rotated_xx[i, j],rotated_yy[i, j], rotated_xx[i+1, j], rotated_yy[i+1, j], rotated_xx[i+1, j+1], rotated_yy[i+1, j+1], rotated_xx[i, j+1], rotated_yy[i, j+1])                                 
            center_points.append(center_point)
            center_points_XX[i,j] = center_point[0]
            center_points_YY[i,j] = center_point[1]
Nbins = len(center_points)



coordsTransform = [5.47956943, 4.16858575]  ### to transfor IDDP-1 onto [0,0]
transX = coordsTransform[0]
transY = coordsTransform[1]


# %% ---------------------------------------- PLOTTING PARAMETERS

### parameters for plotting
imageAlong = 'Image_alongY'  ##'Image_alongY' (S-N) trending cross-sections, or 'Image_alongY' (W-E) trending cross-sections
colorMode = 'k' # 'k' or 'r'
figsize = (120,70)
yLim = [3.5, 1.5]
xLim = [-1,1]
scaFac = 10 ### scaling for wiggle plot
perc_ampl = 99.4 ### scaling for pcolormesh plot

if imageAlong == 'Image_alongY':
    Nlines = Nlines_x
    alongString = ' ALONG Y'

if imageAlong == 'Image_alongX':
    Nlines = Nlines_y
    alongString = 'ALONG X'
#-------------------------------------#

### loop over frequency ranges 
freqRange = [5,16]
freqmin = freqRange[0]
freqmax = freqRange[1]

if imageAlong == 'Image_alongY':
    x_target_offset_list = np.arange(0.6, 1.4, gridIncX)

if imageAlong == 'Image_alongX':
    x_target_offset_list = np.arange(1.2, 1.5, gridIncX)

plotLines = [int(round(x_target_offset/ gridIncX)) for x_target_offset in x_target_offset_list]



# %%

print(str(gridIncX) + ', ' + str(gridIncY) + ', angle = ' + str(float(angle)))


path_results_root = './RESULTS/Wavetype_' + wavetype 
path_results = path_results_root + '/GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle)+'/' 
path_results +=  'NUMPY_MATRICES/'
filename_res =  path_results + '3Dmatrix_freqRange' + str(freqmin) + '-' + str(freqmax) + '.npy'

### load file 
D3matrix = np.load(filename_res)
localCoherFilter = True

for iterLine in plotLines: 
    if imageAlong == 'Image_alongY':
        matrix = copy.deepcopy(D3matrix[:, :, iterLine]) 
        matrix /= (abs(matrix).max())

    if imageAlong == 'Image_alongX':
        matrix = copy.deepcopy(D3matrix[:, iterLine, :])
        matrix /= abs(matrix).max()

    if localCoherFilter: 

        slopes = np.linspace(-200,200, 10)
        slopes = np.linspace(-0.4, 0.4, 5)

        x_positions = np.arange(0, matrix.shape[1], 1)
        data = copy.deepcopy(matrix[100:250, :])
        aper_Z = 7 
        aper_X = 3 

        data_multC = fm.local_coherency_filter(data,
                                x_positions,
                                slopes,
                                aperture_depth=aper_Z,
                                aperture_width=aper_X,
                                semblance=True)

        matrix[100:250,:] = data_multC

    fig = plot_vertical_cross_section(matrix, iterLine, freqRange,scaFac = scaFac, figsize = figsize, yLim = yLim, xLim = xLim, individualEvent = None, perc_ampl = perc_ampl, fold = [], depthIdxFold = 0, imageAlong = imageAlong)
    plt.tight_layout()


