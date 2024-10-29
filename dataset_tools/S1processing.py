import sys
sys.path.append('/home/cyprien/.snap/snap-python')
from esa_snappy import ProductIO, GPF, HashMap, Engine, WKTReader, jpy
import os
import gc
import numpy as np
import rasterio as rio
#from rasterio.enums import Resampling
import shapefile
import geopandas as gpd
 

#config = Engine.getInstance().getConfig()
#config.logLevel('ALL')

# Enable Java garbage collection
gc.enable()
gc.collect()
# Load all operators
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

def process_sentinel1(input_file, output_file, wkt, polarization='VV'):
    System = jpy.get_type('java.lang.System')
    System.gc()
    # Read the input product
    print("Reading product")
    product = ProductIO.readProduct(input_file)
    
    # Apply Orbit File
    print("Apply orbit file")
    parameters = HashMap()
    parameters.put('continueOnFail', True)
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
    product = GPF.createProduct('Apply-Orbit-File', parameters, product)
    
    # Thermal Noise Removal
    print("T Noise removal")
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    product = GPF.createProduct('ThermalNoiseRemoval', parameters, product)

    # Calibration
    print("Calibration")
    parameters = HashMap()
    parameters.put('outputSigmaBand', False)
    parameters.put('sourceBands', f'Intensity_{polarization}')
    parameters.put('selectedPolarisations', polarization)
    parameters.put('outputImageScaleInDb', True)
    product = GPF.createProduct('Calibration', parameters, product)
    
    # Speckle Filtering
    print("Speckle filtering")
    parameters = HashMap()
    parameters.put('filter', 'Lee Sigma')
    parameters.put('windowSize', '5x5')
    product = GPF.createProduct('Speckle-Filter', parameters, product)
    
    # Terrain Correction
    print("TC")
    proj = 'PROJCS["RGF93 v2 / Lambert-93",GEOGCS["RGF93 v2",DATUM["Reseau_Geodesique_Francais_1993_v2",SPHEROID["GRS 1980",6378137,298.257222101],TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","9777"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",46.5],PARAMETER["central_meridian",3],PARAMETER["standard_parallel_1",49],PARAMETER["standard_parallel_2",44],PARAMETER["false_easting",700000],PARAMETER["false_northing",6600000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","9793"]]'
    parameters = HashMap()
    parameters.put('demName', 'SRTM 1Sec HGT')
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('pixelSpacingInMeter', 10.0)
    parameters.put('nodataValueAtSea',False)
    #parameters.put('maskOutAreaWithoutElevation',False)
    parameters.put('mapProjection', proj)
    product = GPF.createProduct('Terrain-Correction', parameters, product)
    
    print("From Linear to Db")
    parameters = HashMap()
    parameters.put('sourceBands', 'Sigma0_VV')
    product = GPF.createProduct('LinearToFromdB', parameters, product)

    # Subset
    print("Subseting")
    parameters = HashMap()
    #parameters.put('sourceBands','Sigma0_VV')
    parameters.put('landMask', True)
    parameters.put('useSRTM', True)
    
    #parameters.put('shorelineExtension', 3.0)
    #parameters.put('geometry', wkt)
    #product = GPF.createProduct('Land-Sea-Mask', parameters, product)
    

    # Write the output product
    print("Writing")
    pm = createProgressMonitor()
    #ProductIO.writeProduct(product, output_file, 'GeoTIFF', pm)
    ProductIO.writeProduct(product, output_file, 'BEAM-DIMAP', pm)

    # Convert uint8

    # Dispose of the product to free up memory
    product.dispose()

def createProgressMonitor():
    PWPM = jpy.get_type('com.bc.ceres.core.PrintWriterProgressMonitor')
    JavaSystem = jpy.get_type('java.lang.System')
    monitor = PWPM(JavaSystem.out)
    return monitor

def convert_float32_tiff_to_uint8(input_file, output_file):
    with rio.open(input_file) as src:
        # Read the image data
        image = src.read(1)  # Assuming single band image
        # Normalize the image to 0-255 range
        min_val = np.min(image)
        max_val = np.max(image)
        normalized = np.round(((image - min_val) / (max_val - min_val) * 255)).astype(np.uint8)
    
        # Create a new profile for the output image
        profile = src.profile.copy()
        profile.update(
            dtype=rio.uint8,
            count=1,
            compress='jpeg',
            driver='JPEG'
        )
    
        # Write the normalized image as JPEG
        with rio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized, 1)
    

# Usage example
input_folder = "/DATA/S1_Med/images"
output_folder = "/DATA/S1_Med/processed"
#shp_path = "/DATA/S1_Med/Med_France_EPSG9793.shp"
#shp_path = "/DATA/S1_Med/test.shp"
# Read the shapefile
#gdf = gpd.read_file(shp_path)
#wkt = gdf['geometry'].to_wkt()
geometry = "POLYGON((1143527.64274951769039035 6106000.0489121600985527, 1217725.72201186302118003 6027568.08727050479501486, 1226459.28158519649878144 5949457.68559567164629698, 1108485.93838328821584582 6065921.5839253319427371, 1108485.93838328821584582 6065921.5839253319427371, 1143527.64274951769039035 6106000.0489121600985527))"
#geometry = WKTReader().read(geometry)
for filename in os.listdir(input_folder):
    if filename.endswith(".zip") and filename.startswith("S1"):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_processed")
        
        process_sentinel1(input_file, output_file, geometry)
                
        #convert_float32_tiff_to_uint8(output_file +'.tif', output_file + '.jpg')
        # Force garbage collection
        gc.collect()

