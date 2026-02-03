# Earthquake migration imaging

This repository provides the Python codes and seismic data used in 'A Shallow Reflector Beneath Krafla Volcano, NE Iceland, Detected With Seismic Migration of Local Earthquake Phases' by Regina Maass, Ka Lok Li, Christopher J. Bean, Benjamin Schwarz and Ivan Lokmer (submitted, https://essopenarchive.org/doi/full/10.22541/essoar.175596898.84807159). 

CODES: Includes the Python codes for the workflow: 
- 1_binning.py: Binning (gridding) of the study area.
- 2_processing.py: Processing of data and stacking in CIP (common-image-point) gathers.
- 3_plotting.py: Data visualization (plotting).
- funtions_migration.py: contains additional functions that are used in 1_binning, 2_processing, and 3_plotting. 

DATA: Contains the seismic data used in the paper:
- stored in .mseed format at a sampling rate of 200 Hz.
- three files are provided per earthquake, corresponding to recordings of station line L1, station line L2, and the array (ARR). A detailed description of the station setup can be found in the manuscript.

META: Contains additional files needed in to reproduce results from the paper:
- station_info.csv: station information (longitude, latitude) for all stations used in the paper. 
- earthquake_info.csv: earthquake information (date, time, longitude, latitude, depth below sea level, magnitude) for all events used in the paper.
- PStraveltimes: pre-calculated traveltimes for direct P- and S waves for each earthquake-station combination. 
- Traveltime_matrices: contains pre-calculated reflection traveltimes and bounce (reflection) points for different reflectors spaced at 20m intervals between ~2 km depth and 6 km depth. For each earthquake, one file exists containing the information for all stations. 

## Description

The RVSP (reverse vertical seismic profiling) method - more commonly used in controlled-source seismology - was here adapted for imaging with earthquakes. The code is demonstrated using local earthquake data from Krafla volcano, NE Iceland, in a Python framework. 

## Support
If you have any questions or comments, please don't hesitate to contact me.

