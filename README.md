# Earthquake_migration_imaging

This repository provides the Python codes used in 'A Shallow Reflector Beneath Krafla Volcano, NE Iceland, Detected With Seismic Migration of Local Earthquake Phases' by Regina Maass, Ka Lok Li, Christopher J. Bean, Benjamin Schwarz and Ivan Lokmer (submitted). 

CODES: Includes the codes for the workflow: 
- 1_binning.py: Binning (gridding) of the study area.
- 2_processing.py: Processing of data and stacking in CIP (common-image-point) gathers.
- 3_plotting.py: Data visualization (plotting).

META: Contains files needed in to reproduce results from the paper
- station_info.csv: station information (longitude, latitude, elevation) for all stations used in the paper. 
- earthquake_info.csv: earthquake information (longitude, latitude, depth, magnitude) for all events used in the paper.
- PStraveltimes: pre-calculated traveltimes for direct P- and S waves for each earthquake-station combination. 
- subfolder Traveltime_matrices: contains pre-calculated reflection traveltimes and bounce (reflection) points for different reflectors spaced at 20m intervals between ~2 km depth and 6 km depth. For each earthquake, one file exists containing the information for all stations. 


## Description

The RVSP (reverse vertical seismic profiling) method - more commonly used in controlled-source seismology - was here adapted for imaging with earthquakes. The code is demonstrated using local earthquake data from Krafla volcano, NE Iceland, in a Python framework. 

## Support
If you have any questions or comments, please don't hesitate to contact maass@cp.dias.ie.


## Project status
This is a new repository and it will be expanded, improved and updated in the next couple of weeks and months. 