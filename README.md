# CSTAT+
## Two-point connectivity statistics computation for hydrological patterns
This project repository includes a new algorithm (CSTAT+) for the computation of two-point connectivity statistics, which is fully vectorized and partially adopted operations of a convolutional neural network. CSTAT+ is designed to quantify the connectivity states for hydrological patterns (e.g., surface flow and soil moisture) in surface topography of high-resolution grid. The hydrological patterns are mostly spatial and temporal snapshots of GIS raster files with total number of pixels at the magnitude of millions to billions, based on the results of physically-based and distributed hydrological models or observations (i.e., in-situ and/or remotely sensed). The benefit of two-point connectivity statistics is to capture changes in hydrosystem states by aggregated metrics that can be directly compared for systems in different hydroclimatic regimes and landscapes. 

## Instructions
“CSTATplus” folder includes two Python scripts (CSTATplus_OMNI.py and CSTATplus_TOPO.py): one for OMNI (2D omnidirectional connectivity) and the other one for TOPO (3D directional connectivity in topography). Please make sure that the required module MXNet, GDAL and Scipy are installed in the Python programming environment, using Pip, Anaconda or compiling from source. 
Note that all of the input files must be in GIS raster formats and the supported file format can be found at https://www.gdal.org/formats_list.html

CSTATplus_OMNI.py: Please change the input work directory and file name at line 347-348 for the input flow pattern. 
CSTATplus_TOPO.py: Please change the input work directory and file name at line 705-706 and 716-717 for the input flow pattern and DEM grids. The input flow pattern and DEM must have identical dimensions (i.e., with the row and column length).

The results will be written to the disk as .csv files, including (for both OMNI and TOPO):
1. <filename> results_taoh.csv
Connectivity probability tao(h) for each lag bin and the aggregated connectivity index OMNI across the separation distances. 
2. <filename> results_CARD.csv
Connectivity probability tao(h) for each lag bin at each of the four spatial orientation in a 2D Euclidean space (i.e., NS, WE, NE-SW or NW-SE) and the aggregated connectivity index TOPO across the separation distances. 
3. <filename> computingtime.csv
Total computational time in seconds.
  
 “Testdata” folder includes all the test datasets (GIS raster files in geotiff format) used in the manuscript. 
  
## License
CSTAT+ is distributed under the Apache License Version 2.0. The license is available in source directory or online: http://www.apache.org/licenses/LICENSE-2.0

## References
A research article describing CSTAT+ along with a text cases was published in Environmental Modeling & Software:

Yu, F., & Harbor, J. M., 2019. CSTAT+: A GPU-accelerated spatial pattern analysis algorithm for high-resolution 2D/3D hydrologic connectivity using array vectorization and convolutional neural network operations. Environmental Modelling & Software, 104496. https://doi.org/10.1016/j.envsoft.2019.104496

A case study for the CSTAT+ implementation of connectivity computation and hydrological implication was published in Hydrological Processes:

Yu, F., Harbor, J. M., 2019. The effects of topographic depressions on multiscale overland flow connectivity: A high‐resolution spatiotemporal pattern analysis approach based on connectivity statistics. Hydrol. Process. (33) 1403–1419. https://doi.org/https://doi.org/10.1002/hyp.13409

For the original concept and algorithm of connectivity statistics that CSTAT+ is based, please see:

Western, A.W., Blöschl, G., Grayson, R.B., 2001. Toward capturing hydrologically significant connectivity in spatial patterns. Water Resources Research (37) 83–97. https://doi.org/10.1029/2000WR900241


