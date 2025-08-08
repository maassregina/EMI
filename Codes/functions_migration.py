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
    evLat = df.Latitude
    evLon = df.Longitude
    evDepth = df.Depth
    evMag = df.Magnitude

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



