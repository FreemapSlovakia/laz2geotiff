# laz2geotiff

Build a **seamless DEM GeoTIFF** of the Czech Republic from `*.laz` files, using
Natural Neighbour Interpolation (NNI) over a Delaunay TIN (`startin`).

Seams are avoided by an **interpolation margin**: each output tile is rendered at
its exact final pixel dimensions, but the points fed into the TIN are queried
from the tile bbox *plus* `--margin` meters (pulled from a sqlite bbox index).
Because NNI is local, edge pixels get the same value they'd have in a
country-wide triangulation, so a `gdalbuildvrt` mosaic has no seams.

Output tiles are Float32 GeoTIFF, Deflate-compressed, georeferenced to the
source CRS (no reprojection). Final near-lossless **LERC_ZSTD** compression and
clipping to the state border are applied in the GDAL step at the end.

## Pipeline for `~/14TB/DEM/cz/laz`

Assumes the release binary; from the crate dir it's
`target/release/laz2geotiff` (aliased `L` below).

```sh
L=~/fm/laz2geotiff/target/release/laz2geotiff
CZ=~/14TB/DEM/cz
```

### 1. Index the laz files (bbox index; done once)

```sh
$L index $CZ/laz $CZ/cz-index.sqlite
```

### 2. Render the DEM tiles (parallel, resumable)

```sh
$L render \
  --index      $CZ/cz-index.sqlite \
  --out-dir    $CZ/tiles \
  --resolution 1.0 \
  --epsg       5514 \
  --tile-size  4096 \
  --margin     50 \
  --max-edge   0 \
  --jobs       24
```

- `--surface all` is the default and correct here: these files store the terrain
  as class 8, so any classification filter would drop everything. Use `--dtm`
  (class 2) or `--dsm` only for standard-classified data.
- Resumable: re-run the same command to continue; already-written tiles are
  skipped (tiles are written atomically via a `.tif.tmp` rename).
- `--max-edge <m>`: mask pixels whose enclosing TIN triangle has an edge longer
  than `<m>` — trims interpolation across the concave exterior and big empty
  gaps. **It also holes water bodies wider than `<m>`**, so leave it at `0` and
  let the cutline (step 4) define the border precisely; only raise it if the
  pre-clip convex-hull fill bothers you.
- Memory scales with `tile-size²` × point density; 4096 @ ~0.06 pts/m² is a few
  hundred MB per thread. Lower `--tile-size` or `--jobs` if RAM is tight.

### 3. Build the seamless VRT mosaic

Too many tiles for shell globbing, so use a file list:

```sh
find $CZ/tiles -name 'dem_*.tif' > $CZ/tiles.list
gdalbuildvrt -input_file_list $CZ/tiles.list $CZ/cz_dem.vrt
```

### 4. Single lossless BigTIFF, then clip to the border

Cutline from ČÚZK RÚIAN (state polygon, EPSG:5514):
`$CZ/cutline/STATY_P.shp`
(downloaded from <https://services.cuzk.gov.cz/shp/stat/epsg-5514/1.zip>, layer `STATY_P`).

`STATY_P` is the **exact** administrative border, but the laz data overshoots it
(flight-line overlap; measured median ~300 m, up to ~1 km along the border). To
keep that useful cross-border data while still cropping the far convex-hull
fill, clip to a **dilated + simplified** border.

**Why not `gdalwarp -cutline -crop_to_cutline`?** It re-rasterizes the whole
cutline polygon *per warp region* (`GDALWarpCutlineMaskerEx` → `GDALRasterizeGeometries`),
single-threaded. With the 77k-vertex border it pins one core at 100 % with **zero
I/O and no progress** — effectively hung. Decouple instead: write the file, then
rasterize the cutline once.

```sh
# one-time: buffer the border outward by 500 m, then SIMPLIFY (77k -> ~5.5k verts).
# Simplification is what makes the clip fast; invisible at a 500 m buffer / 1 m grid.
ogr2ogr -f "ESRI Shapefile" $CZ/cutline/STATY_P_buf500.shp $CZ/cutline/STATY_P.shp \
  -dialect SQLITE -sql "SELECT ST_Buffer(geometry, 500) AS geometry FROM STATY_P"
ogr2ogr -f "ESRI Shapefile" $CZ/cutline/STATY_P_buf500_s25.shp \
  $CZ/cutline/STATY_P_buf500.shp -simplify 25
```

```sh
# 4a. VRT -> one lossless BigTIFF (ZSTD+predictor 3). ~1 h; ZSTD-9 compression is
#     the pacing limiter (uses ~8 cores), not disk. ZSTD-3 is much faster, ~same size.
gdal_translate $CZ/cz_dem.vrt $CZ/cz_dem.tif \
  --config GDAL_CACHEMAX 2048 \
  -co COMPRESS=ZSTD -co ZSTD_LEVEL=9 -co PREDICTOR=3 \
  -co TILED=YES -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co SPARSE_OK=TRUE

# 4b. Clip: burn nodata OUTSIDE the border, in place (single pass, fits without bloat).
#     Single-threaded (~2 h for CZ) -- the slow part. gdal_rasterize can't multithread.
gdal_rasterize -i -burn -9999 -l STATY_P_buf500_s25 \
  $CZ/cutline/STATY_P_buf500_s25.shp $CZ/cz_dem.tif
```

Note: LERC was dropped — benchmarked lossless, ZSTD+predictor 3 (34 M sample) beats
LERC_ZSTD `MAX_Z_ERROR=0` (62 M); LERC can't use the float predictor.

### 5. (optional) Overviews for fast viewing

```sh
gdaladdo -r average \
  --config GDAL_NUM_THREADS ALL_CPUS \
  --config COMPRESS_OVERVIEW ZSTD --config PREDICTOR_OVERVIEW 3 \
  $CZ/cz_dem.tif  2 4 8 16 32 64 128
```

## `xyz`: a terrain-tiles source straight from points

Renders on the EPSG:3857 XYZ pixel grid of `--zoom` as Int32 heights in
`--step` metres (nodata `-2147483648`, ZSTD + predictor 2, 256 px blocks), so
the result needs no warp — only a mosaic and overviews.

```sh
$L xyz --index sk-index.sqlite --ct ct_sk.txt --out-dir blocks \
  --zoom 20 --block-zoom 13 --tile-zoom 17 --margin 40 --max-edge 30 --classes 2
```

- `--ct` is a file holding the PROJ pipeline from the index's CRS to EPSG:3857.
  It is evaluated exactly on a `--grid` (100 m) lattice and interpolated
  bilinearly in between (sub-millimetre error), with z = 0 like `gdalwarp -ct`.
- Only point formats 6+ are read (selective LAZ decompression of XYZ + class).
- The points of one `--block-zoom` tile (plus margin) are held in RAM, 12 B
  each; each file is re-read for every block it overlaps, so bigger blocks
  read less but need more RAM.
- Each `--tile-zoom` tile is triangulated with `--margin` metres of neighbours
  and rasterised linearly; triangles with an edge over `--max-edge` stay
  nodata. Tiles agree on their shared edges within one step.
- One GeoTIFF per block, renamed into place when complete; `blocks.done` lists
  finished blocks, empty ones included. Re-run the same command to resume.
