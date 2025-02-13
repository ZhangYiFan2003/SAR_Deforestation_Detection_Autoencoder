import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import numpy as np

def fuse_and_split_images(
    vv_dir,
    vh_dir,
    fused_dir,
    tiles_dir,
    tile_size=256,
    prefix_fused="fused",
    prefix_tile="tile"
):
    """
    Fuse VV and VH images into a multi-band GeoTIFF, then split the fused large image into complete georeferenced tiles (only saving 256x256 tiles).
    
    :param vv_dir:       Directory containing VV images
    :param vh_dir:       Directory containing VH images
    :param fused_dir:    Output directory for the fused images
    :param tiles_dir:    Output directory for the split tiles
    :param tile_size:    Tile size in pixels, default is 256
    :param prefix_fused: Prefix for the fused files, default is "fused"
    :param prefix_tile:  Prefix for the tile files, default is "tile"
    """
    
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    
    # Retrieve all TIFF files in the VV and VH directories
    vv_files = sorted([f for f in os.listdir(vv_dir) if f.endswith('.tif')])
    vh_files = sorted([f for f in os.listdir(vh_dir) if f.endswith('.tif')])
    
    assert len(vv_files) == len(vh_files), "The number of files in the VV and VH directories do not match"
    
    for vv_file, vh_file in zip(vv_files, vh_files):
        vv_path = os.path.join(vv_dir, vv_file)
        vh_path = os.path.join(vh_dir, vh_file)
        
        # Filename for the fused image
        fused_filename = f"{os.path.splitext(vv_file)[0]}.tif"
        fused_path = os.path.join(fused_dir, fused_filename)
        
        # Check if the fusion has already been done to avoid duplication
        if not os.path.exists(fused_path):
            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                # Check if CRS and transform are consistent
                if (vv_src.crs != vh_src.crs) or (vv_src.transform != vh_src.transform):
                    raise ValueError(f"The CRS or Transform of files {vv_file} and {vh_file} are inconsistent")
                
                # Check if dimensions are consistent
                if (vv_src.width != vh_src.width) or (vv_src.height != vh_src.height):
                    raise ValueError(f"The sizes of files {vv_file} and {vh_file} do not match")
                
                # Read VV and VH data (assuming single band for each)
                vv_data = vv_src.read(1)  
                vh_data = vh_src.read(1)  
                
                # Stack into 2 bands
                fused_data = np.stack([vv_data, vh_data], axis=0)
                
                # Define output metadata
                fused_meta = vv_src.meta.copy()
                fused_meta.update({
                    'count': 2,  # Two bands
                    'dtype': fused_data.dtype
                })
                
                # Write the fused multi-band GeoTIFF
                with rasterio.open(fused_path, 'w', **fused_meta) as dst:
                    dst.write(fused_data)
                
                print(f"Fused and saved: {fused_path}")
        else:
            print(f"Fused file already exists: {fused_path}")
        
        # Split the fused image into tiles
        with rasterio.open(fused_path) as fused_src:
            src_transform = fused_src.transform
            src_crs = fused_src.crs
            src_width = fused_src.width
            src_height = fused_src.height
            band_count = fused_src.count  # Should be 2
            
            # Calculate the number of complete tiles
            num_tiles_row = src_height // tile_size
            num_tiles_col = src_width // tile_size
            
            for tile_row in range(num_tiles_row):
                for tile_col in range(num_tiles_col):
                    row_off = tile_row * tile_size
                    col_off = tile_col * tile_size
                    
                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=tile_size,
                        height=tile_size
                    )
                    
                    # Calculate the affine transform for this tile
                    tile_transform = window_transform(window, src_transform)
                    
                    # Read data for the tile
                    tile_data = fused_src.read(
                        indexes=list(range(1, band_count + 1)),
                        window=window
                    )
                    
                    # Define the tile filename
                    tile_filename = f"{os.path.splitext(fused_filename)[0]}_{row_off}_{col_off}_fused.tif"
                    tile_path = os.path.join(tiles_dir, tile_filename)
                    
                    # Define tile metadata
                    tile_meta = fused_src.meta.copy()
                    tile_meta.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': tile_transform
                    })
                    
                    # Write the tile
                    with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                        dst.write(tile_data)
                    
                    print(f"Tile split and saved: {tile_path}")
    
    print("All images have been successfully fused and split into tiles.")

if __name__ == "__main__":
    vv_dir = '/home/yifan/Documents/data/forest/test/VV'
    vh_dir = '/home/yifan/Documents/data/forest/test/VH'
    fused_dir = '/home/yifan/Documents/data/forest/test/fused'
    tiles_dir = '/home/yifan/Documents/data/forest/test/processed'
    fuse_and_split_images(vv_dir, vh_dir, fused_dir, tiles_dir, tile_size=256, prefix_fused="fused", prefix_tile="tile")