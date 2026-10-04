# laz2geotiff

Builds seamless digital terrain models from airborne lidar point clouds (`*.laz`).
It is how [Freemap](https://www.freemap.sk)'s terrain sources for
[terrain-tiles](https://github.com/FreemapSlovakia/terrain-tiles) are made
from national lidar sets, which come as thousands of flight-line files.

## Commands

| Command | What it does |
| --- | --- |
| `index` | Scans a directory of `*.laz` files into a sqlite index of their bounding boxes. |
| `xyz` | Renders the points straight onto the Web Mercator XYZ pixel grid of one zoom, as Int32 height steps, one GeoTIFF per block. This is the source format terrain-tiles reads. |
| `fill` | Fills the gaps an `xyz` render leaves, seamlessly across blocks. |
| `render` | The older path: grids the points in the source CRS as Float32 tiles plus a VRT. |

`laz2geotiff <command> --help` lists every option.

## `xyz`

Points are reprojected through a PROJ pipeline (`--ct`, a file holding the
pipeline from the source CRS to EPSG:3857, sampled on a bilinear grid of exact
transforms), triangulated per tile of `--tile-zoom` plus `--margin`, and each
pixel is interpolated by natural neighbours (Sibson) over the Delaunay
triangulation.

- **Seamless without a global triangulation.** A tile sees the points of its
  margin too, so neighbouring tiles see the same triangles near their shared
  edge and interpolate it identically.
- **Gaps.** A triangle whose longest edge is over `--gap-cap` (EPSG:3857
  metres), or over `--gap-factor` times the local median edge, is left as
  nodata for `fill`; natural neighbours would bulge across it. `--gap-factor 0`
  keeps only the absolute cap. The margin must exceed the cap, or two tiles may
  triangulate a gap differently.
- **Blocks.** Work goes in blocks of `--block-zoom`. A block's points are read
  once and held in RAM, then its tiles render in parallel into one file. The
  next block's points are read on their own `--read-jobs` threads while the
  current one renders. A block file appears only once complete and is recorded
  in `blocks.done`, so an interrupted run resumes where it stopped.
- **Reading.** A flight line is far longer than a block, so the first read of a
  file records each LAZ chunk's box (`chunks.sqlite` in the output directory),
  and later blocks decode only the chunks that reach them. Only x, y, z and the
  class are decoded, and only `--classes` are kept.
- **Memory.** Before a tile starts, a gate estimates its need from its point
  count and waits while that would leave less than `--mem-reserve` GB of the
  system's available memory. `<out-dir>/limits`, holding `<jobs> <reserve GB>`,
  overrides both and is re-read before every tile, so a running render can be
  throttled without a restart.

Heights are stored in `--step` metres (2 mm by default), with the scale set on
the band, and nodata is `i32::MIN`.

## `fill`

A membrane (Laplace) fill. A coarse field of the whole mosaic at
`--coarse-zoom` supplies each block window's outer edge, so a gap spanning
several blocks, such as a reservoir, is filled as one surface. A gap enclosed
by data is filled whole; one open to the edge of the data only within
`--reach` metres of it.

Blocks are filled independently, so their edges can disagree slightly. A second
pass with `--boundary` (a VRT of the first pass's output) takes each window's
edge from that result instead, and the seams disappear.

## Example: Slovakia at z18

```sh
L=target/release/laz2geotiff

$L index /data/sk/laz sk-index.sqlite

# Ground points only; natural neighbours everywhere except gaps over 100 m.
$L xyz --index sk-index.sqlite --ct ct_sk.txt --out-dir blocks \
  --zoom 18 --block-zoom 13 --tile-zoom 16 --classes 2 \
  --margin 120 --gap-factor 0 --gap-cap 100 --jobs 24 --read-jobs 8

gdalbuildvrt blocks.vrt blocks/*.tif
$L fill --input blocks.vrt --blocks blocks --out-dir fill1
gdalbuildvrt fill1.vrt fill1/*.tif
mkdir -p fill2 && cp fill1/coarse_z12.f32 fill2/   # reuse the coarse field
$L fill --input blocks.vrt --blocks blocks --out-dir fill2 --boundary fill1.vrt
```

The filled blocks are then clipped to a cutline, merged into one GeoTIFF with
`gdal_translate`, and given overviews with `gdaladdo -r average`.

## Building

```sh
cargo build --release
```

It builds with jemalloc and for the native CPU by default (`.cargo/config.toml`),
so the binary will not run on an older CPU than the one it was built on. Needs
GDAL and PROJ.

## Licence

GPL-3.0-or-later; see [LICENSE](LICENSE).
