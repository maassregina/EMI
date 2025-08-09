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

import obspy 
from obspy import read, Trace, Stream
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.image as mpimg
matplotlib.rc('font', size=25)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import matplotlib.patches as patches
import matplotlib.image as img 
import source_autocorrelations as sa

import vtk
import pyvista as pv
import numpy as np
from scipy.signal import hilbert


def local_coherency_filter(data,
                           x_positions,
                           slopes,
                           aperture_depth=11,
                           aperture_width=7,
                           semblance=True):

    from scipy.ndimage import shift

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



def matrix_preprocessing(matrix, AgControl = False, normByMax = False, envelopes = False, derivative = None, whitenInDepthDomain = False, timeGate = 1, integrate = False):
    ### PRE-PROCESSING OF MATRIX 

    sampRate = 250 #create artificial sampling rate 
    npts = matrix.shape[0]
    timeVec = np.arange(0, npts/sampRate, 1/sampRate)

    # # AGC if needed
    # if AgControl:
    #     timeGate = 1
    #     matrix = sf.agc(matrix, timeVec, agc_type = 'rms',  time_gate = timeGate ) #500
        

    if envelopes:
        from scipy.signal import hilbert
        matrix_new = np.zeros(matrix.shape)
       # hilbertTransform = hilbert(np.real(matrix))
        #for row in matrix:
        for iterRow in range(matrix.shape[1]):
            row = matrix[:, iterRow]
            if np.all((row == 0)) == False:  # Check if the row contains non-zero data
                envelope = np.abs(hilbert(row))  # Calculate envelope using Hilbert transform
                envelope = tikhonov_smooth(envelope, alpha=1)

                matrix_new[:, iterRow] = envelope

            else:
                matrix_new[:, iterRow] = row

            
        matrix = matrix_new
        # Convert the result back to a numpy array
        #matrix_new = np.array(matrix_new)
    

    if (derivative == 'second') or (derivative == 'first'):
       # npts = stream[0].stats.npts
       # sampRate = stream[0].stats.sampling_rate
      #  t = np.arange(0, npts/sampRate, 1/sampRate)
        N = matrix.shape[0]
        z = np.linspace(reflectorDepthsAll[0], reflectorDepthsAll[-1], N)

        env = np.zeros((matrix.shape))

        if derivative == 'second':
            for iterTr in range(matrix.shape[1]):
                env_second_der =   np.gradient(np.gradient(matrix[:,iterTr], z), z)
                env_second_der = tikhonov_smooth(env_second_der, alpha=3)

                env[:, iterTr] = env_second_der

        elif derivative == 'first':
            for iterTr in range(matrix.shape[1]):
                first_der =   np.gradient(matrix[:, iterTr], z)
                first_der = tikhonov_smooth(first_der, alpha=3)

                env[:, iterTr] = first_der
        matrix = env

    # normalize by max val of each bin
    if normByMax:
        for iterTrace in range(matrix.shape[1]):
            maxVal = max(abs(matrix[:, iterTrace]))
            if maxVal > 10e-15:
                matrix[:, iterTrace] /= maxVal

    if integrate: 
        for iterTrace in range(matrix.shape[1]):
            matrix[:, iterTrace] = np.cumsum(matrix[:, iterTrace])
    
    for iterTrace in range(matrix.shape[1]):
        matrix[:,iterTrace] - np.mean(matrix[:,iterTrace])

    return matrix


def plot_grid(iterLine, imageAlong, ax1, individualEvent = None, fold = [], depthIdxFold = 0, cmap = 'Greys', plotEarthquakes = True, coordsTransform = [5.47956943, 4.16858575], angle = -7):
    ### Plotting
    mapname = 'GridIncr' + str(gridIncX) + '-' + str(gridIncY)

    transX = coordsTransform[0]
    transY = coordsTransform[1]

    # gs = gridspec.GridSpec(4, 2, figure=fig)
    # ax1 = fig.add_subplot(gs[0:4, 0])

    ax1.plot(stXall - transX, stYall - transY,'ko', alpha = 0.6, markersize = 60)
    ax1.plot(stXInclude - transX, stYInclude - transY, "ko", alpha = 1, markersize = 60)

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
       # cax = fig.add_axes([0.065, 0.94, 0.1, 0.02])  #[x, y, width, height] for the colorbar
      #  cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
       # cbar.ax.set_xlabel('Number')  # Add a label to the colorbar



   # ax1.plot(center_points_XX, center_points_YY, 'go', markersize=20, zorder =-1)

   # plt.gca().set_aspect('equal', adjustable='box')
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

        x_rect_rot, y_rect_rot = rotate_point((x_rect, y_rect), angle, refCoords = (xRef, yRef))
        ax1.fill(x_rect_rot - transX, y_rect_rot - transY, color='b', alpha=0.4, label='Shaded Area')  # Fill area


    if imageAlong == 'Image_alongX':
        ymin = rotated_yy_angle0[iterLine][0]
        ymax = rotated_yy_angle0[iterLine+1][0]
        xmin = np.min(rotated_xx_angle0)
        xmax = np.max(rotated_xx_angle0)

        x_rect = [xmin, xmax, xmax, xmin]
        y_rect = [ymin, ymin, ymax, ymax]

        x_rect_rot, y_rect_rot = rotate_point((x_rect, y_rect), angle, refCoords = (xRef, yRef))
      #  ax1.fill_betweenx([ymin, ymax], xmin, xmax, color='green', alpha=1)
        ax1.fill(x_rect_rot - transX, y_rect_rot - transY, color='b', alpha=0.4,edgecolor = 'k', label='Shaded Area')  # Fill area
        

    return ax1


def plot_vertical_cross_section(matrix, iterLine, freqRange, scaFac = 1, figsize = None, yLim = None, xLim = None, individualEvent = None, perc_ampl = 97, fold = [], depthIdxFold = 0, imageAlong = 'imageAlong_Y'):
    
    
    matplotlib.rc('font', size=150)
    matplotlib.rc('xtick', labelsize=120) 
    matplotlib.rc('ytick', labelsize=120) 

    freqmin = freqRange[0]
    freqmax = freqRange[1]

    Nlines = matrix.shape[1]
    if figsize == None:
        figsize = (40,15)
    fig = plt.figure(figsize=figsize)
  #  if Nlines <= 35:
    gs = gridspec.GridSpec(4, 10, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    ax = fig.add_subplot(gs[0:4, 4:7])
    ax3 = fig.add_subplot(gs[0:4, 7:10], sharey = ax)


    # if Nlines > 35:
    #     gs = gridspec.GridSpec(4, 9, figure=fig)
    #     ax = fig.add_subplot(gs[0:4, 0:4])
    #     ax1 = fig.add_subplot(gs[0:4, 5:9])

    ### create x axes
    if imageAlong == 'Image_alongX':
        # p0 = rotated_xx[0][0]
        # addInc = gridIncX
        # pos_vline = xIDDP
        xLim_list = center_points_XX[iterLine,:]

    if imageAlong == 'Image_alongY':
        # p0 = rotated_yy[0][0]
        # addInc = gridIncY
        # pos_vline = yIDDP
        xLim_list = center_points_YY[:, iterLine]
    
       # p1 = rotated_xx[-1]

       # X_ticks = np.arange(p0, p1, gridIncX)

    ax1 = plot_grid(iterLine, imageAlong, ax1, individualEvent = individualEvent, fold = fold, depthIdxFold=depthIdxFold)
    ax1.set_xlim(3.8, 6.6)
    ax1.set_ylim(2.6, 5.6)

    ax1.set_xlim(map_limits[0])
    ax1.set_ylim(map_limits[1])
    # ax1.set_xlim(4.7, 6)
    # ax1.set_ylim(3, 5.6)
   # ax1.gca().set_aspect('equal', adjustable='box')
    ax1.plot(xIDDP1, yIDDP1, marker = 'o', color = 'yellow', markeredgecolor = 'k', markeredgewidth = 10, markersize = 115)
   
    #count = p0
    #xLim_list = []

    for iterBin in range(Nlines):
        count = xLim_list[iterBin]
        amplitudes = matrix[:, iterBin]

        if len(amplitudes) == 0: 
          #  ax.plot(np.nan, np.nan + count)
            ax.plot(count, np.nan, 'X', lw = 2)

        else:
    # amplitudes /= max(abs(amplitudes)) 
            amplitudes = amplitudes/scaFac
            amplitudes += count

            ax.plot(amplitudes, reflectorDepthsAll, '-', c = 'k')
            if colorMode == 'k':
                ax.fill_betweenx(reflectorDepthsAll,count,amplitudes,where=(amplitudes>count),color='k')

            elif colorMode == 'r':
                ax.fill_betweenx(reflectorDepthsAll,count,amplitudes,where=(amplitudes>count),color='r')
                ax.fill_betweenx(reflectorDepthsAll,count,amplitudes,where=(amplitudes<count),color='b')
        
        # xLim_list.append(count)
        # count += addInc
        




    # #xTicks = np.arange(0, count//2, 1)


    #plt.ylim(max(depths),min(depths))
    #aplt.xticks(np.arange(0, count, 2), xTicks, fontsize = 15, rotation = 90)
    #plt.xlim(50, 200)
    YTICKS = np.arange(min(reflectorDepthsAll), max(reflectorDepthsAll), 0.5)

    # x_rect = [xmin, xmax, xmax, xmin]
    # y_rect = [ymin, ymin, ymax, ymax]
    # x_rect_rot, y_rect_rot = rotate_point((x_rect, y_rect), angle, refCoords = (xRef, yRef))

    ax.set_yticks(YTICKS)        


    #ax.text(xIDDP, 1.5, 'IDDP1', fontsize = 50)
    #plt.xticklabels(binsPlot)


    #plt.xlim(xLim[0], xLim[1])
    # #plt.xlim(200)
    # plt.grid()

  #  ax.axvline(pos_vline, c= 'orange', ls = '--', lw = 14)
    ax.set_title('Fequency range ' + str(freqmin) + '-' + str(freqmax) + ' Hz')

    ###
    X = np.arange(0, matrix.shape[1], 1)
    Y = reflectorDepthsAll
    
    cmap = 'Greys'

    matrix[np.isnan(matrix)] = 0
    matrix[~np.isfinite(matrix)] = 0
    vm=np.percentile(np.absolute(matrix), perc_ampl)

    im = ax3.pcolormesh(xLim_list, Y, matrix, cmap = cmap, vmin = -vm, vmax =vm)
   # fig.colorbar(im, ax=ax3, orientation = 'horizontal')

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

# ax1.text(xIDDP+ 0.04, yIDDP - 0.038, 'IDDP1', fontsize = 50)
    plt.suptitle(imageAlong + ', line no.' + str(iterLine) + ' - Stacked Earthquakes')

    #fig.tight_layout()

    return fig



import scipy.signal as signal
import scipy.fftpack as fft

def return_matrix_fold(freqRange, pathResultsROOT, postproc_stringROOT, newMethod = False):
        ### load path according to freqUency range
        freqmin = freqRange[0]
        freqmax = freqRange[1]

        filterPost = True
        if freqmin == None:
            filterPost = False
        if filterPost == True: 
            postproc_string = '_freqRange' + str(freqmin) + '-' + str(freqmax) + postproc_stringROOT
        if filterPost == False:
            postproc_string = '_noFilter' + postproc_stringROOT


        # if synthetics == True:
        if path_earthquakes.split('/')[4].endswith('synthetics'):
            synthString = path_earthquakes.split('/')[5]
            postproc_string = '_synthetics_' + synthString + '_calcEnvelopes' + str(calcEnvelopes) + '_derivative-' + str(derivative)
            postproc_string += '_normalize_' + str(normalize_stream)
            postproc_string += '_maskDir_' + str(maskDirectWaves)

        results_string = '3Dmatrix' + postproc_string

        if newMethod:
            results_string = '3Dmatrix_newMethod' + postproc_string
        print(results_string)

        fold_string = 'Fold' + postproc_string


        pathResults_ = pathResultsROOT + '/POSTPROCESSING/' 
        pathResults = pathResults_ + results_string
        pathResultsFold = pathResults_ + fold_string


        ### load data
       # print('... load file')
        file = pathResults +  '.npy'
        fileFold = pathResultsFold +  '.npy'

        D3matrix = np.load(file)
        D3fold = np.load(fileFold)

        return D3matrix, D3fold


def plot_figure_for_line_freqRange(iterLine, freqRange, thresholdFold = None, imageAlong = 'Image_alongY', plotFold = False, depthIdxFold = 0):
        
        D3matrix, D3fold = return_matrix_fold(freqRange, pathResultsROOT, postproc_stringROOT)

       # Set the corresponding values in matrix2 to 0
        if thresholdFold:
            indices = D3fold < thresholdFold
            D3matrix[indices] = 0

        ### extract some variables
        Ndep = D3matrix.shape[0]
        Nlines_y = D3matrix.shape[1]
        Nlines_x = D3matrix.shape[2]


        if imageAlong == 'Image_alongY':
            matrix = copy.deepcopy(D3matrix[:, :, iterLine]) ## I think that's correct
            matrix /= (abs(matrix).max())

        if imageAlong == 'Image_alongX':
            matrix = copy.deepcopy(D3matrix[:, iterLine, :])
            matrix /= abs(matrix).max()

        ### preprocess matrix
        matrix = matrix_preprocessing(matrix,AgControl = AgControl, normByMax = normByMax,  envelopes = envelopes,derivative = deriv,whitenInDepthDomain = whitenInDepthDomain, timeGate = tGate, integrate = integrate)
        if im_multwithCOHER:
            matrix = multiplywithCOHER(matrix, stackFacCoher = im_coherStackFac)
        if whitenInDepthDomain: 
            fs = 50
            matrix = spectral_whitening(matrix, fs, fmin=whiten_freqmin, fmax=whiten_freqmax, epsilon=1e-10)

        matrix /= abs(matrix).max()

        if AgControl:
            timeGate = timeGateImageDomain
            dt = (reflectorDepthsAll[1]-reflectorDepthsAll[0])
            timeVec = np.arange(0, matrix.shape[0]/(1/dt),dt)
            matrix = sf.agc(matrix, timeVec, agc_type = 'rms',  time_gate = timeGate ) #500
            

        ### call plotting function
        fold = []
        if plotFold == True: 
            fold = D3fold
        fig = plot_vertical_cross_section(matrix, iterLine, freqRange, scaFac=scaFac, figsize = figsize, yLim = yLim, xLim = xLim, perc_ampl = perc_ampl, fold = fold, depthIdxFold = depthIdxFold)
        fig.tight_layout()

        return fig, matrix, D3fold, D3matrix


#### DEFINE SUBMATRIX TO BE PLOTTED 

### first column: deoth
### second column: Y-index - Imaging along LONGITUDE (the x-axis)
### third column: X-index - Imaging along LATITUDE (the y-axis)



# %%########################################################################################
############################################################################################
####-------------------------------- DEFINE CRUCIAL VARIABLES
############################################################################################
############################################################################################
############################################################################################
#### define phase to be analysed.
wavetype = 'RP'

#### print statements? 
verbose = True


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

### grid increment - can be several (e.g., [0.1, 0.07]), code will loop over grid increments
gridIncX= 0.2 #spacing in x direction, dx
gridIncY = 0.2 #spacing in y direction, dy

### rotation angle of the grid  - can be several (e.g., [-7, 40]), code will loop over grid increments)
angle = -7

# %% ------ GRIDDING


### Create rotated grid.
gridLengths = [gridIncX, gridIncY]
xRef, yRef = sf.LonLatToKm([lonIDDP], [latIDDP], coordsRefLonLat = [lonRef, latRef])
map_limits = [[-0.72745307497261036 + xRef, 0.7386664214407499 + xRef],
        [0.61930828928136261] + yRef, -0.81598651190213167 + yRef]
map_limits = [[4.75, 6.05], [3.3, 4.8]]
rotated_xx, rotated_yy = fm.create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(xRef, yRef))
rotated_xx_angle0, rotated_yy_angle0 = fm.create_rotated_grid(modelSpaceLimits, gridLengths, 0, refCoords =(xRef, yRef))

### extract boundaries 
x0 = rotated_xx[0][0]
x1 = rotated_xx[0][-1]
y0 = rotated_yy[0][0]
y1 = rotated_yy[0][-1]        

######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
Nlines_x = rotated_xx.shape[1] - 1
Nlines_y = rotated_xx.shape[0] -1
NbinsAll = (Nlines_x * Nlines_y)
offsets_x = np.linspace(x0, x1,Nlines_x)

plotLines = [int(round(x_target_offset/ gridIncX)) for x_target_offset in x_target_offset_list]

center_points = [] 
center_points_XX = np.zeros((Nlines_y, Nlines_x))
center_points_YY = np.zeros((Nlines_y, Nlines_x))

######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
for i in range(Nlines_y):
        for j in range(Nlines_x):
            center_point = fm.compute_center_point(rotated_xx[i, j],rotated_yy[i, j], rotated_xx[i+1, j], rotated_yy[i+1, j], rotated_xx[i+1, j+1], rotated_yy[i+1, j+1], rotated_xx[i, j+1], rotated_yy[i, j+1])                                 
            center_points.append(center_point)
            center_points_XX[i,j] = center_point[0]
            center_points_YY[i,j] = center_point[1]
Nbins = len(center_points)


if imageAlong == 'Image_alongY':
    Nlines = Nlines_x
    alongString = ' ALONG Y'

if imageAlong == 'Image_alongX':
    Nlines = Nlines_y
    alongString = 'ALONG X'




# %%########################################################################################
############################################################################################
####----------------- PLOTTING PARAMETERS AND PROCESSING IN IMAGE DOMAIN
############################################################################################
############################################################################################
############################################################################################


### preprocessing of matrix?
AgControl = False
timeGateImageDomain = 0.5
normByMax = False
envelopes = False
whitenInDepthDomain = False
whiten_freqmin = 0.8
whiten_freqmax = 6
deriv = 'None'
tGate = 0.8
im_coherStackFac = 4
im_multwithCOHER = False
if im_multwithCOHER == False:
    im_coherStackFac = None
integrate = False

### parameters for plotting
imageAlong = 'Image_alongY'
scaFac = 0.3
colorMode = 'k'
figsize = (25,20)
#if Nlines > 35:
    # figsize = (35,24)
figsize = (120,70)
yLim = [3.5, 1.5]
# yLim = [6,1.5]
xLim = [3.7,4.9]
xLimY = [4.2, 5]
xLimX = [4, 5]

#xLim = None
perc_ampl = 99.4 ### for pcolormesh plot

#-------------------------------------#

### loop over frequency ranges 
freqRanges = [[None, None]]

coordsTransform = [5.47956943, 4.16858575]
transX = coordsTransform[0]
transY = coordsTransform[1]

gridIncX = 0.2
gridIncY = 0.2
angle = -7


freqRange = [5,16]
freqmin = freqRange[0]
freqmax = freqRange[1]

if imageAlong == 'Image_alongY':
    x_target_offset_list = np.arange(0.7, 1, gridIncX_list[0])

if imageAlong == 'Image_alongX':
    x_target_offset_list = np.arange(1.05, 1.5, gridIncX_list[0])


scaFac = 8


xLim3 = [3.5, 5]
xLim2 = [3.2, 4.7]
xLim1 = [3, 4.5]

thresholdFold = 1

localCoherFilter = True

stacks_per_line = []

matrices_all = []


# %%

print(str(gridIncX) + ', ' + str(gridIncY) + ', angle = ' + str(float(angle)))


path_results_root = './RESULTS/Wavetype_' + wavetype 
path_results = path_results_root + '/GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle)+'/' 
path_results +=  'NUMPY_MATRICES/'
filename_res =  path_results + '3Dmatrix_freqRange' + str(freqmin) + '-' + str(freqmax) + '.npy'

### load file 
matrix = np.load(filename_res)

perc_ampl = 94
xLim = 2.7,5.3
fig = plot_vertical_cross_section(matrix, iterLine, freqRange, scaFac = 1, figsize = figsize, yLim = None, xLim = xLim, individualEvent = None, perc_ampl = perc_ampl, fold = [], depthIdxFold = 0)
plt.tight_layout()
plt.ylim(3.5,1.8)


# %%######### ------------ VERTICAL CROSS-SECTIONS: LOOP OVER FREQUENCY RANGES AND LINES
#########################################################
#########################################################
#########################################################
#########################################################

for iterLine in plotLines: 
        if imageAlong == 'Image_alongX':
            xLim = [4.9,6.5]
        if imageAlong == 'Image_alongY':
            xLim = [2.5,4.5]
        fig, matrix, fold, D3matrix = plot_figure_for_line_freqRange(iterLine, freqRange, thresholdFold=thresholdFold, imageAlong = imageAlong, plotFold = plotFold, depthIdxFold = depthIdx)

# %% 
            #plt.close()
        #  plt.savefig('/home/maass/Desktop/colorbar.svg', format = 'SVG')




# %% ------------- PLOTTING WITH PYVISTA 
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

### some necessary steps 
#####################################################################        
gridIncX = 0.07
gridIncY = 0.07
gridLengths = [gridIncX, gridIncY]
xRef, yRef = sf.LonLatToKm([lonIDDP1], [latIDDP1], coordsRefLonLat = [lonRef, latRef])
rotated_xx_angle0, rotated_yy_angle0 = create_rotated_grid(modelSpaceLimits, gridLengths, 0, refCoords =(xRef, yRef))

rotated_xx, rotated_yy = copy.deepcopy(rotated_xx_angle0), copy.deepcopy(rotated_yy_angle0)
map_limits = [[4.75 - transX, 6.05 - transX], [3.3 - transY, 4.8 - transY]]

### extract boundaries 
x0 = rotated_xx[0][0]
x1 = rotated_xx[0][-1]
y0 = rotated_yy[0][0]
y1 = rotated_yy[0][-1]        

######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
Nlines_x = rotated_xx.shape[1] - 1
Nlines_y = rotated_xx.shape[0] -1
NbinsAll = (Nlines_x * Nlines_y)
offsets_x = np.linspace(x0, x1,Nlines_x)


center_points = [] 
center_points_XX = np.zeros((Nlines_y, Nlines_x))
center_points_YY = np.zeros((Nlines_y, Nlines_x))

######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
for i in range(Nlines_y):
        for j in range(Nlines_x):
            center_point = compute_center_point(rotated_xx[i, j],rotated_yy[i, j], rotated_xx[i+1, j], rotated_yy[i+1, j], rotated_xx[i+1, j+1], rotated_yy[i+1, j+1], rotated_xx[i, j+1], rotated_yy[i, j+1])                                 
            center_points.append(center_point)
            center_points_XX[i,j] = center_point[0]
            center_points_YY[i,j] = center_point[1]
Nbins = len(center_points)

##########################################################################

center_points_XX = np.zeros((Nlines_y, Nlines_x))
center_points_YY = np.zeros((Nlines_y, Nlines_x))

######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
for i in range(Nlines_y):
        for j in range(Nlines_x):
            center_point = compute_center_point(rotated_xx[i, j],rotated_yy[i, j], rotated_xx[i+1, j], rotated_yy[i+1, j], rotated_xx[i+1, j+1], rotated_yy[i+1, j+1], rotated_xx[i, j+1], rotated_yy[i, j+1])                                 
            center_points.append(center_point)
            center_points_XX[i,j] = center_point[0]
            center_points_YY[i,j] = center_point[1]
Nbins = len(center_points)


import numpy as npW
import pyvista as pv

def closest_number_index(num_list, target):
    closest = min(num_list, key=lambda x: abs(x - target))
    return num_list.index(closest)

coordsTransform = [5.47956943, 4.16858575]
transX = coordsTransform[0]
transY = coordsTransform[1]

CutvalX_left = 0.7
CutvalX_right = 0.7

CutvalY_down = 0.8
CutvalY_up = 0.6

for iter in range(Nlines_x):
    if center_points_XX[0, iter] - transX > -(CutvalX_left + gridIncX):
        print(iter,center_points_XX[0, iter]- transX)
        break
iterX_cut = iter
left_corner_x = center_points_XX[0, iter]  - transX

for iter in range(Nlines_x):
    if center_points_XX[0, iter] - transX> CutvalX_right:
        print(iter,center_points_XX[0, iter]- transX)
        break
iterX_cut_ = iter

# right_corner_x = center_points_XX[iter, iter]

for iter in range(Nlines_y):
    if center_points_YY[iter,0] - transY> -CutvalY_down:
        print(iter,center_points_YY[iter, 0]- (transY + gridIncX))
        break
iterY_cut = iter
left_corner_y = center_points_YY[iter, 0]  - transY

for iter in range(Nlines_y):
    if center_points_YY[iter,0] - transY> CutvalY_up:
        print(iter,center_points_YY[iter, 0]- transY)
        break
iterY_cut_ = iter

dy = center_points_YY[2, 0] - center_points_YY[1, 0]
dx = center_points_XX[0, 2] - center_points_XX[0,1]

#right_corner_y = center_points_YY[iter, iter]

#### extract depth range
depth_start = 1.8
depthIdx_start = closest_number_index(reflectorDepthsAll, depth_start)

depth_end = 3.6 #3.6, 4.5
depthIdx_end = closest_number_index(reflectorDepthsAll, depth_end)


# %%  -------------- LOAD 2D MATRICES FOR CERTAIN ANGLES AND PLOT 
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


def figure_combineAngles(x_target_offsets, figsize = (60,60), saveFig = False, pathSaveFig = '/home/maass/Desktop/', yLim = [3.5,1.5]):
    Nsections = len(x_target_offsets)

    fig = plt.figure(figsize=figsize)
    #gs = gridspec.GridSpec(4, len(plotLines) +1, figure=fig)
    gs = gridspec.GridSpec(2, Nsections, figure=fig)


    for iterSection in range(Nsections):
        imageAlong = imagesAlong[iterSection]
        scaFacPlot = scaFacPlots[iterSection]
        angle = angle_list[iterSection]
        x_offset = x_target_offsets[iterSection]

        gridIncX = gridIncX_list[0]
        gridIncY = gridIncY_list[0]

        if imageAlong == 'Image_alongY':
           # x_target_offset_list = np.arange(1.3, 1.8, gridIncX_list[0])
            x_target_offset_list = np.arange(1.5, 1.9, gridIncX_list[0])

        if imageAlong == 'Image_alongX':
            x_target_offset_list = np.arange(0.7, 1.4, gridIncX_list[0])
            x_target_offset_list = np.arange(0.9, 1.3, gridIncX_list[0])

        #####################################################################                
        print(str(gridIncX) + ', ' + str(gridIncY) + ', angle = ' + str(float(angle)))

        angle = float(angle)
        if angle == -7 or angle == 0 or angle == -20 or angle == -10 or angle == -3:
            angle = int(angle)

        pathResultsROOT = pathResults_ROOT_all + 'GridIncr' + str(gridIncX) + '-' + str(gridIncY) + '_angle' + str(angle) 
        if testdataset == True:
            pathResultsROOT += '/TestDataSet'
            pathResultsROOT += preproc_string

    #  print('... results from' + pathResultsROOT.split('MIGRATION_NEW/')[1])

        ### Create rotated grid.
        gridLengths = [gridIncX, gridIncY]
        xRef, yRef = sf.LonLatToKm([lonIDDP1], [latIDDP1], coordsRefLonLat = [lonRef, latRef])
        map_limits = [[-0.72745307497261036 + xRef, 0.7386664214407499 + xRef],
                [0.61930828928136261] + yRef, -0.81598651190213167 + yRef]
        map_limits = [[4.75, 6.05], [3.3, 4.8]]
        if setting == 'Controlled_sources':
            map_limits = [[3.5, 7], [2.5, 6]]
        rotated_xx, rotated_yy = create_rotated_grid(modelSpaceLimits, gridLengths, angle, refCoords =(xRef, yRef))
        rotated_xx_angle0, rotated_yy_angle0 = create_rotated_grid(modelSpaceLimits, gridLengths, 0, refCoords =(xRef, yRef))



        map_limits = [[4.75 - transX, 6.05 - transX], [3.3 - transY, 4.8 - transY]]
    
        ### extract boundaries 
        x0 = rotated_xx[0][0]
        x1 = rotated_xx[0][-1]
        y0 = rotated_yy[0][0]
        y1 = rotated_yy[0][-1]        

        ######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
        Nlines_x = rotated_xx.shape[1] - 1
        Nlines_y = rotated_xx.shape[0] -1
        NbinsAll = (Nlines_x * Nlines_y)
        offsets_x = np.linspace(x0, x1,Nlines_x)
        
        plotLines = [int(round(x_target_offset/ gridIncX)) for x_target_offset in x_target_offset_list]

        center_points = [] 
        center_points_XX = np.zeros((Nlines_y, Nlines_x))
        center_points_YY = np.zeros((Nlines_y, Nlines_x))

        ######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
        for i in range(Nlines_y):
                for j in range(Nlines_x):
                    center_point = compute_center_point(rotated_xx[i, j],rotated_yy[i, j], rotated_xx[i+1, j], rotated_yy[i+1, j], rotated_xx[i+1, j+1], rotated_yy[i+1, j+1], rotated_xx[i, j+1], rotated_yy[i, j+1])                                 
                    center_points.append(center_point)
                    center_points_XX[i,j] = center_point[0]
                    center_points_YY[i,j] = center_point[1]
        Nbins = len(center_points)


        if imageAlong == 'Image_alongY':
            Nlines = Nlines_x
            alongString = ' ALONG Y'

        if imageAlong == 'Image_alongX':
            Nlines = Nlines_y
            alongString = 'ALONG X'




        #####################################################################
        #### AVERAGING OVER FREQUENCY RANGES

        D3matrixAv = []
        for iterFreq in range(len(freqRanges)): 
            freqRange = freqRanges[iterFreq]
            D3matrix, D3fold = return_matrix_fold(freqRange, pathResultsROOT, postproc_stringROOT)

            if smooth_D3matrices:
                kernel_size = (kz, ky, kx)
                D3matrix = uniform_filter(D3matrix, size=kernel_size, mode='nearest')


            if thresholdFold:
                indices = D3fold < thresholdFold
                D3matrix[indices] = 0
            
            try: 
                D3matrixAv += D3matrix
            except: 
                D3matrixAv = D3matrix

        D3matrixAv /= len(freqRanges)
        D3matrix = copy.deepcopy(D3matrixAv)
        D3matrices.append(D3matrix)

        #####################################################################
        ############################ EXTRACT 2D MATRIX AND PREPROCESS IT
        iterLine = int(x_offset/gridIncX)
        lines_idx.append(iterLine)

        ################## FIND CENTRE POINT THAT IS CLOSEST TO  LINE
        center_points

        if imageAlong == 'Image_alongY':
            matrix = copy.deepcopy(D3matrix[:, :, iterLine]) ## I think that's correct
            matrix /= (abs(matrix).max())
            center_points_line_x = center_points_XX[:, iterLine] - transX
            center_points_line_y = center_points_YY[:, iterLine] - transY
            y_closest_zero = center_points_line_y[np.argmin(np.abs(center_points_line_y))]
            x_closest_zero = center_points_line_x[np.argmin(np.abs(center_points_line_y))]

            points_cutx_x.append(x_closest_zero)
            points_cutx_y.append(y_closest_zero)

            axLabel = 'Y [km]'
            xLim = [-1, 1]

        if imageAlong == 'Image_alongX':
            matrix = copy.deepcopy(D3matrix[:, iterLine, :])
            matrix /= abs(matrix).max()
            center_points_line_x = center_points_XX[iterLine, :] - transX
            center_points_line_y = center_points_YY[iterLine,:] - transY
          #  x_closest_zero = center_points_line_x[np.argmin(np.abs(center_points_line_x))]
            y_closest_zero = center_points_line_y[np.argmin(np.abs(center_points_line_x))]
            x_closest_zero = center_points_line_x[np.argmin(np.abs(center_points_line_x))]

      #      y_closest_zero = center_points_line_y[np.argmin(np.abs(center_points_line_y))]
 
            points_cutx_x.append(x_closest_zero)
            points_cutx_y.append(y_closest_zero)

      #      points_y.append(y_closest_zero)

            axLabel = 'X [km]'
            xLim = [-0.8, 0.6]
            xLim = [-0.6, 1]

        #### preprocessing of matrix 
        matrix = matrix_preprocessing(matrix,AgControl = AgControl, normByMax = normByMax,  envelopes = envelopes,derivative = deriv,whitenInDepthDomain = whitenInDepthDomain, timeGate = tGate, integrate = integrate)
        if im_multwithCOHER:
            matrix = multiplywithCOHER(matrix, stackFacCoher = im_coherStackFac)
        
        if localCoherFilter: 

            slopes = np.linspace(-200,200, 10)
            slopes = np.linspace(-0.4, 0.4, 5)
            #slopes = np.linspace(-1000, 1000, 50)

            x_positions = np.arange(0, matrix.shape[1], 1)
            data = copy.deepcopy(matrix[100:250, :])
            aper_Z = 7 #5
            aper_X = 3 # must be odd, 7

            data_multC = local_coherency_filter(data,
                                    x_positions,
                                    slopes,
                                    aperture_depth=aper_Z,
                                    aperture_width=aper_X,
                                    semblance=True)


        #  matrix_enh = copy.deepcopy(matrix)
            matrix[100:250,:] = data_multC

     #   matrix /= np.max()
        matrices_all.append(matrix)
        #####################################################################
        ########## PLOTTING

        scaFac = scaFacPlot


        axMap = fig.add_subplot(gs[0, iterSection])
        ax = fig.add_subplot(gs[1, iterSection])

                                #### MAP
        fold = []
        depthIdxFold = 0
        axMap = plot_grid(iterLine, imageAlong, axMap, fold = fold, depthIdxFold=depthIdxFold, angle = -7)
        axMap.set_xlim(3.8, 6.6)
        axMap.set_ylim(2.6, 5.6)
        axMap.set_title('')

        axMap.set_xlim(map_limits[0])
        axMap.set_ylim(map_limits[1])
        axMap.plot(xIDDP1 - transX, yIDDP1 - transY, marker = 'o', color = 'yellow', markeredgecolor = 'k', markeredgewidth = 10, markersize = 115)
    

        #### now line plot
        ax = plot_vertical_cross_section_mod(ax, matrix, iterLine, freqRange, scaFac = scaFac, figsize = figsize, yLim = None, xLim = xLim, individualEvent = None, perc_ampl = 97, fold = [], depthIdxFold = 0, imageAlong = imageAlong)
        ax.set_ylim(yLim[0], yLim[1])
        ax.set_xlim(xLim[0], xLim[1])

        if iterFreq != len(freqRanges) - 1:
            plt.setp(ax.get_xticklabels(), visible=False);
        else: 
            ax.set_xlabel(axLabel)
        if iterSection != 0:
            plt.setp(ax.get_yticklabels(), visible=False);
            plt.setp(axMap.get_yticklabels(), visible=False);
            axMap.set_ylabel('')
        else: 
            ax.set_ylabel( 'Depth [km]')

    plt.tight_layout()
    if saveFig:
        plt.savefig(pathSaveFig + 'Cross_sections_' + imagesAlong[0]+ '_' + str(x_target_offsets[0]) +  '.png', format = 'PNG', dpi = 30)
        plt.savefig(pathSaveFig + 'Cross_sections_' + imagesAlong[0]+ '_' + str(x_target_offsets[0]) +  '.svg', format = 'SVG', dpi = 30)



def plot_pyvista_fig(plotSections = [0,1], saveFig = False, pathSavePlot = '/home/maass/Desktop/'):
    import numpy as np
    import pyvista as pv
    import vtk

    # vmin = -1
    # vmax = 1
    center_points_XX = np.zeros((Nlines_y, Nlines_x))
    center_points_YY = np.zeros((Nlines_y, Nlines_x))

    ######### -1 because the rotated_xx -and yy values are the BOUNDARIES of the grid cells, and the center points related to the actual number of bins
    for i in range(Nlines_y):
            for j in range(Nlines_x):
                center_point = compute_center_point(rotated_xx_angle0[i, j],rotated_yy_angle0[i, j], rotated_xx_angle0[i+1, j], rotated_yy_angle0[i+1, j], rotated_xx_angle0[i+1, j+1], rotated_yy_angle0[i+1, j+1], rotated_xx_angle0[i, j+1], rotated_yy_angle0[i, j+1])                                 
                center_points.append(center_point)
                center_points_XX[i,j] = center_point[0] - transX
                center_points_YY[i,j] = center_point[1] - transY
    Nbins = len(center_points)


    ### CREATE EMPTY 3D MATRICES 
    nx = D3matrices[0].shape[0]
    ny = D3matrices[0].shape[1]
    nz = D3matrices[0].shape[2]

    matrix3D_1= np.zeros((nx, ny, nz))
    thetas = []

    slices = pv.MultiBlock()  # treat like a dictionary/list

    for iterSection in plotSections:
        if imagesAlong[iterSection] == 'Image_alongY':
            matrix3D_1[:, :, lines_idx[iterSection]] = matrices_all[iterSection]
            theta = 0
            xpos = center_points_XX[0,:][lines_idx[iterSection]]
            ypos = 0
        if imagesAlong[iterSection] == 'Image_alongX':
            matrix3D_1[:, lines_idx[iterSection], :] = matrices_all[iterSection]
            theta = 90
            xpos = 0
            ypos = center_points_YY[:,0][lines_idx[iterSection]]

        ### here: compute locations of x and y for plotting:

        center_points_line_x = center_points_XX[lines_idx[iterSection], :] - transX
        center_points_line_y = center_points_YY[lines_idx[iterSection],:] - transY



        #------------- HERE !!!
        # angle_list = [-15, 90 - 10]
        # angle_list = [angle_list[0], 90 - angle_list[1]]
        #angle_list_plot = [0 +angle_list[0], 90 + angle_list[1]]
        angle_list_plot = angle_list



        p_xs = xpos
        p_ys = ypos


        localCoherFilter = False


        angle = angle_list_plot[iterSection]
        data = copy.deepcopy(D3matrices[iterSection])
    #  data = np.where(data == 0, np.nan, data)
        data = copy.deepcopy(matrix3D_1)

      #  theta = thetas[iterSection]

        data = np.transpose(data, axes=(2, 1, 0))


        nx, ny, nz = data.shape
        data = data[iterX_cut: iterX_cut_, iterY_cut: iterY_cut_, depthIdx_start:depthIdx_end]

        data = np.flip(data, axis = 2)

        grid = pv.ImageData()
        grid.dimensions = data.shape



        grid.origin = (left_corner_x, left_corner_y, -depth_end)  # The bottom left corner of the data set
        grid.spacing = (dx,dy, 0.02)  # These are the cell sizes along each axis

        # Add the data values to the cell data
        #grid.point_data['values'] = data.flatten(order='F')  # Flatten the array
        grid.point_data['values'] = data.flatten(order='F')  # Flatten the array

        axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
        model = grid

        theta = theta *(np.pi/180)

        point = np.array(grid.center)
        point[0] = p_xs#[iterSection]
        point[1] = p_ys#[iterSection]
        print(point)

        normal = np.array([np.cos(theta), np.sin(theta), 0.0]).dot(np.pi / 2.0)
        name = f'Angle: {np.rad2deg(angle):.2f}'
        name = str(angle) + '-' + str(theta)
        slic = model.slice(origin=point, normal=normal)
        slices[name] = slic

        slices[name] = slices[name].rotate_z(angle, point=axes.origin, inplace=False)
        # slices

        #slices= slices.rotate_z(-20, point=axes.origin, inplace=False)

    p = pv.Plotter(window_size=[1200,2000])
    pv.global_theme.font.size = 30
    pv.global_theme.font.family = 'arial'
    p.set_background('white')

    #flipped = slices.transform(transform)
    p.add_mesh(slices, cmap=cmap,scalars='values',clim = [vmin,vmax], lighting=True, ambient = 0.4)
    p.add_mesh(model.outline())
    p.show_grid()
    p.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        n_xlabels=8,
        n_ylabels=8,
        n_zlabels=8,
        xtitle='X [km]',
        ytitle='Y [km]',
        ztitle='Depth [km]',
        fmt='%.1f')

    p.camera_position = [(4, 3, 1.25),
    (0.01900249719619751, 0.003252655267715454, -2.6399999856948853),
    (0.0, 0.0, 1.0)]
    p.camera_position = 'xz'  # or 'xz', 'yz'

    for azimuth in azimuths:
        # pathSavePlot = '/home/maass/Desktop/tmp/RESULTS_MIG/D3FIG/' + saveString + '/'
        # if os.path.exists(pathSavePlot)==False:
        #     os.makedirs(pathSavePlot)

        saveString = 'Ang' + str(angle_list) + '_Pos' + str(x_target_offsets[iterSection])
        saveString = saveString  + '_localCoherFilter' + str(localCoherFilter)

        p.camera.azimuth = azimuth

        p.camera.elevation = 15
        p.add_title('AZMIUTH' + str(azimuth))


        p.show()
        saveString += '_AZI' + str(azimuth)
        print(saveString)

        if saveFig:
            p.save_graphic(pathSavePlot + saveString +'.svg')
            p.save_graphic(pathSavePlot+ saveString +'.eps')









# %%

#!HERE!!!
saveFig = False
#pathSaveFig = '/home/maass/Desktop/tmp/RESULTS_MIG/FINAL/'
pathSavePlot = '/home/maass/Desktop/Paper2/MIG_RES/'



localCoherFilter = True
if 'synthetics' in path_earthquakes:
    localCoherFilter = False

combineAngles = True

gridIncX_list = [0.07]
gridIncY_list = [0.07]

freqRanges = [[5,25]]


x_target_offsets = [1.15, 1.27] ### best
x_target_offsets = [1.05, 1.27] ### No 1 - best
x_target_offsets = [1.41, 1.5] ### No 1 - best

x_target_offsets = np.arange(1.54, 1.93, gridIncX_list[0])
x_target_offsets = np.arange(1.13, 1.54, gridIncX_list[0])
#x_target_offsets = np.arange(1.54, 1.93, gridIncX_list[0])
x_target_offsets = [1.41, 1.5] ### No 1 - best

imagesAlong = ['Image_alongX']*len(x_target_offsets)
angle_list = [-7]*len(x_target_offsets) ###No 3

Nsections = len(angle_list)

D3matrices = []

thresholdFold = 1
### initializ()e figure 

scaFacPlots = [2]*len(x_target_offsets)
matrices_all = []
D3matrices = []
points_cutx_x = []
points_cutx_y = []

points_y = []

smooth_D3matrices = False
kz = 2
kx = 2
ky = 2

lines_idx = []
figsize = (120,50)
yLim = [3.6,1.8]

if combineAngles:
    figure_combineAngles(x_target_offsets, figsize = figsize, saveFig = True, pathSaveFig = pathSavePlot, yLim = yLim)






# %% ----
#### PYVISTA FIGURE
azimuths = [0]
saveFig =True

import cmcrameri.cm as cm
cmap = cm.vik
vmin = -0.2
vmax = 0.2

vmin = -0.6 #default
vmax = 0.6 #default

vmin = -0.6
vmax = 0.6

for iterSection in range(Nsections):
    plot_pyvista_fig(plotSections = [iterSection], saveFig= saveFig, pathSavePlot = pathSavePlot)

# %%

