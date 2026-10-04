//! laz2geotiff — seamless DEMs from *.laz point clouds.
//!
//! - `index`  — scan a directory of *.laz files into a sqlite bbox index.
//! - `xyz`    — natural neighbours straight onto the Web Mercator XYZ grid,
//!   one Int32 GeoTIFF per block (see `xyz.rs`).
//! - `fill`   — membrane-fill the gaps `xyz` leaves, seamlessly across blocks.
//! - `render` — the older path: Float32 tiles in the source CRS (default
//!   EPSG:5514) plus a VRT, the margin making adjacent tiles agree.

mod fill;
mod xyz;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use las::{point::Classification, Reader};
use rayon::prelude::*;
use rusqlite::Connection;
use walkdir::WalkDir;
use wbgeotiff::{Compression, GeoTiffWriter, GeoTransform, SampleFormat, WriteLayout};

#[derive(Parser)]
#[command(about = "Build a seamless DEM GeoTIFF from *.laz files")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan a directory of *.laz files into a sqlite bbox index.
    Index(IndexArgs),
    /// Render a tiled DEM (+ list of tiles for a VRT) from the index.
    Render(RenderArgs),
    /// Render an Int32 DTM on the EPSG:3857 XYZ pixel grid, one file per block.
    Xyz(xyz::XyzArgs),
    /// Fill the gaps an `xyz` render leaves, seamlessly across its blocks.
    Fill(fill::FillArgs),
}

#[derive(Parser)]
struct IndexArgs {
    /// Directory containing *.laz files (searched recursively).
    directory: PathBuf,
    /// Output sqlite index file.
    database: PathBuf,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Surface {
    /// Use every point, no classification filter (correct for data that is
    /// already a filtered terrain surface, e.g. this CZ set, all class 8).
    All,
    /// Ground-classified points only (class 2 bare-earth DTM).
    Dtm,
    /// All points except water/noise (surface DSM).
    Dsm,
}

#[derive(Parser)]
struct RenderArgs {
    /// Sqlite index built by `index`.
    #[arg(long)]
    index: PathBuf,
    /// Output directory for the DEM tiles.
    #[arg(long)]
    out_dir: PathBuf,
    /// Grid resolution in CRS units (meters) per pixel.
    #[arg(long, default_value_t = 1.0)]
    resolution: f64,
    /// EPSG code of the laz data (also the output CRS; no reprojection).
    #[arg(long, default_value_t = 5514)]
    epsg: u32,
    /// Tile size in pixels (square). Bigger = fewer laz re-reads but more RAM.
    #[arg(long, default_value_t = 1024)]
    tile_size: u32,
    /// Interpolation margin in meters: extra points loaded around each tile so
    /// edges have their true natural neighbours (must exceed point spacing).
    #[arg(long, default_value_t = 50.0)]
    margin: f64,
    /// Which points feed the surface.
    #[arg(long, value_enum, default_value_t = Surface::All)]
    surface: Surface,
    /// Max triangle edge length in meters; pixels whose enclosing TIN triangle
    /// has a longer edge become NoData. Cuts interpolation across the concave
    /// exterior and large empty gaps. 0 = disabled (fill whole convex hull).
    /// NOTE: also holes water bodies wider than this value, so keep it above the
    /// largest lake/reservoir you want filled (final clipping is via a cutline).
    #[arg(long, default_value_t = 0.0)]
    max_edge: f64,
    /// NoData value written for pixels outside the data.
    #[arg(long, default_value_t = -9999.0)]
    nodata: f64,
    /// Number of worker threads (tiles processed in parallel).
    #[arg(long, default_value_t = default_jobs())]
    jobs: usize,
    /// Optional bbox limit "minx,miny,maxx,maxy" in CRS units (default: full index extent).
    #[arg(long)]
    bbox: Option<String>,
}

fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Index(args) => run_index(args),
        Command::Render(args) => run_render(args),
        Command::Xyz(args) => xyz::run(args),
        Command::Fill(args) => fill::run(args),
    }
}

// ---------------------------------------------------------------------------
// index
// ---------------------------------------------------------------------------

fn run_index(args: IndexArgs) -> Result<()> {
    let files: Vec<PathBuf> = WalkDir::new(&args.directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "laz").unwrap_or(false))
        .map(|e| e.into_path())
        .collect();

    eprintln!("Indexing {} files", files.len());

    // Read headers in parallel; only the bounds are needed.
    let rows: Vec<(f64, f64, f64, f64, String)> = files
        .par_iter()
        .filter_map(|path| {
            let reader = Reader::from_path(path).ok()?;
            let b = reader.header().bounds();
            Some((
                b.min.x,
                b.max.x,
                b.min.y,
                b.max.y,
                path.to_string_lossy().into_owned(),
            ))
        })
        .collect();

    let mut conn = Connection::open(&args.database)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS laz_index (min_x REAL, max_x REAL, min_y REAL, max_y REAL, file TEXT)",
        (),
    )?;

    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare("INSERT INTO laz_index VALUES (?1, ?2, ?3, ?4, ?5)")?;
        for r in &rows {
            stmt.execute((r.0, r.1, r.2, r.3, &r.4))?;
        }
    }
    tx.commit()?;

    for q in [
        "CREATE UNIQUE INDEX IF NOT EXISTS laz_file_unique ON laz_index (file)",
        "CREATE INDEX IF NOT EXISTS laz_min_x_index ON laz_index (min_x)",
        "CREATE INDEX IF NOT EXISTS laz_max_x_index ON laz_index (max_x)",
        "CREATE INDEX IF NOT EXISTS laz_min_y_index ON laz_index (min_y)",
        "CREATE INDEX IF NOT EXISTS laz_max_y_index ON laz_index (max_y)",
    ] {
        conn.execute(q, ())?;
    }

    eprintln!("Indexed {} files into {}", rows.len(), args.database.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

/// floor division for i64 (Rust `/` truncates toward zero).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn run_render(args: RenderArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out_dir)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build_global()
        .ok();

    let res = args.resolution;
    let tsz = args.tile_size as i64;

    // Overall extent (from index or --bbox), in CRS units.
    let (ext_min_x, ext_min_y, ext_max_x, ext_max_y) = {
        let conn = Connection::open(&args.index)?;
        if let Some(b) = &args.bbox {
            let v: Vec<f64> = b.split(',').map(|s| s.trim().parse().unwrap()).collect();
            anyhow::ensure!(v.len() == 4, "--bbox needs minx,miny,maxx,maxy");
            (v[0], v[1], v[2], v[3])
        } else {
            conn.query_row(
                "SELECT min(min_x), min(min_y), max(max_x), max(max_y) FROM laz_index",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .context("empty index?")?
        }
    };

    // Global pixel grid aligned to CRS origin (0,0):
    //   col c covers x in [c*res, (c+1)*res); center = (c+0.5)*res
    //   row r covers y in [-(r+1)*res, -r*res); center = -(r+0.5)*res
    let col_min = (ext_min_x / res).floor() as i64;
    let col_max = (ext_max_x / res).ceil() as i64 - 1;
    let row_min = (-ext_max_y / res).floor() as i64; // north -> small row
    let row_max = (-ext_min_y / res).ceil() as i64 - 1;

    let ti_min = floor_div(col_min, tsz);
    let ti_max = floor_div(col_max, tsz);
    let tj_min = floor_div(row_min, tsz);
    let tj_max = floor_div(row_max, tsz);

    let mut tiles = Vec::new();
    for tj in tj_min..=tj_max {
        for ti in ti_min..=ti_max {
            tiles.push((ti, tj));
        }
    }

    eprintln!(
        "Extent x[{ext_min_x:.1},{ext_max_x:.1}] y[{ext_min_y:.1},{ext_max_y:.1}] \
         -> {} tiles of {}px @ {res} m, {} threads",
        tiles.len(),
        args.tile_size,
        args.jobs
    );

    let done = AtomicU64::new(0);
    let written = AtomicU64::new(0);
    let total = tiles.len() as u64;

    tiles.par_iter().for_each(|&(ti, tj)| {
        match render_tile(&args, ti, tj) {
            Ok(true) => {
                written.fetch_add(1, Ordering::Relaxed);
            }
            Ok(false) => {}
            Err(e) => eprintln!("ERROR tile {ti}_{tj}: {e:#}"),
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 50 == 0 || d == total {
            eprintln!("  {d}/{total} tiles ({} written)", written.load(Ordering::Relaxed));
        }
    });

    // Emit a file list for gdalbuildvrt.
    let list_path = args.out_dir.join("tiles.txt");
    let mut names: Vec<String> = std::fs::read_dir(&args.out_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tif"))
        .collect();
    names.sort();
    std::fs::write(&list_path, names.join("\n") + "\n")?;

    eprintln!(
        "Done: {} tiles written. VRT list: {}",
        written.load(Ordering::Relaxed),
        list_path.display()
    );
    Ok(())
}

/// Returns Ok(true) if a tile file was written, Ok(false) if skipped (empty).
fn render_tile(args: &RenderArgs, ti: i64, tj: i64) -> Result<bool> {
    let res = args.resolution;
    let tsz = args.tile_size as i64;
    let w = args.tile_size as usize;
    let h = args.tile_size as usize;

    let out_path = args.out_dir.join(format!("dem_{ti}_{tj}.tif"));
    if out_path.exists() {
        return Ok(false); // resume: already done
    }

    // Tile world bbox (exact, final pixel grid; no cropping).
    let x_min = (ti * tsz) as f64 * res;
    let x_max = ((ti + 1) * tsz) as f64 * res;
    let y_max = -((tj * tsz) as f64) * res;
    let y_min = -(((tj + 1) * tsz) as f64) * res;

    // Buffered query bbox for point loading.
    let m = args.margin;
    let (qminx, qminy, qmaxx, qmaxy) = (x_min - m, y_min - m, x_max + m, y_max + m);

    // Files overlapping the buffered bbox.
    let files: Vec<String> = {
        let conn = Connection::open_with_flags(
            &args.index,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let mut stmt = conn.prepare(
            "SELECT file FROM laz_index WHERE max_x >= ?1 AND min_x <= ?3 AND max_y >= ?2 AND min_y <= ?4",
        )?;
        let rows = stmt.query_map([qminx, qminy, qmaxx, qmaxy], |r| r.get::<_, String>(0))?;
        rows.filter_map(|r| r.ok()).collect()
    };
    if files.is_empty() {
        return Ok(false);
    }

    // Load points within the buffered bbox and insert into the TIN.
    let mut dt = startin::Triangulation::new();
    let mut any_in_tile = false;
    for file in &files {
        let mut reader = match Reader::from_path(file) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("WARN cannot open {file}: {e}");
                continue;
            }
        };
        for point in reader.points() {
            let p = match point {
                Ok(p) => p,
                Err(_) => continue,
            };
            if !surface_keeps(args.surface, p.classification) {
                continue;
            }
            if p.x < qminx || p.x > qmaxx || p.y < qminy || p.y > qmaxy {
                continue;
            }
            if p.x >= x_min && p.x < x_max && p.y >= y_min && p.y < y_max {
                any_in_tile = true;
            }
            let _ = dt.insert_one_pt(p.x, p.y, p.z);
        }
    }

    // Nothing inside the actual tile -> nothing to render.
    if !any_in_tile || dt.number_of_vertices() < 3 {
        return Ok(false);
    }

    // Pixel-center coordinates, row-major (top row first), matching GDAL layout.
    let mut coords = Vec::with_capacity(w * h);
    for y in 0..h {
        let cy = y_max - (y as f64 + 0.5) * res;
        for x in 0..w {
            let cx = x_min + (x as f64 + 0.5) * res;
            coords.push([cx, cy]);
        }
    }

    let nni = startin::interpolation::NNI { precompute: true };
    let zs = startin::interpolation::interpolate(&nni, &mut dt, &coords);

    let nodata = args.nodata as f32;
    let max_edge_sq = args.max_edge * args.max_edge;
    let mut data = vec![nodata; w * h];
    let mut any_valid = false;
    for (i, z) in zs.into_iter().enumerate() {
        let v = match z {
            Ok(v) => v,
            Err(_) => continue, // outside convex hull -> NoData
        };
        // Mask pixels sitting on an over-long triangle (concave exterior / gaps).
        if args.max_edge > 0.0 {
            let [cx, cy] = coords[i];
            match dt.locate(cx, cy) {
                Ok(tri) if triangle_max_edge_sq(&dt, &tri) > max_edge_sq => continue,
                Ok(_) => {}
                Err(_) => continue,
            }
        }
        data[i] = v as f32;
        any_valid = true;
    }
    if !any_valid {
        return Ok(false);
    }

    write_geotiff(args, &out_path, w, h, x_min, y_max, data)?;
    Ok(true)
}

/// Longest squared 2D edge of a triangle, by vertex indices into the TIN.
fn triangle_max_edge_sq(dt: &startin::Triangulation, tri: &startin::Triangle) -> f64 {
    let p: Vec<Vec<f64>> = tri
        .v
        .iter()
        .map(|&vi| dt.get_point(vi).unwrap_or_else(|_| vec![0.0, 0.0, 0.0]))
        .collect();
    let d2 = |a: &[f64], b: &[f64]| {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        dx * dx + dy * dy
    };
    d2(&p[0], &p[1]).max(d2(&p[1], &p[2])).max(d2(&p[2], &p[0]))
}

fn surface_keeps(surface: Surface, c: Classification) -> bool {
    match surface {
        Surface::All => true,
        Surface::Dtm => c == Classification::Ground,
        Surface::Dsm => !matches!(
            c,
            Classification::Water | Classification::LowPoint | Classification::HighNoise
        ),
    }
}

fn write_geotiff(
    args: &RenderArgs,
    path: &Path,
    w: usize,
    h: usize,
    origin_x: f64,
    origin_y: f64,
    data: Vec<f32>,
) -> Result<()> {
    let epsg: u16 = args
        .epsg
        .try_into()
        .with_context(|| format!("epsg {} does not fit in u16", args.epsg))?;

    // North-up transform: origin is the top-left corner, negative pixel height.
    let gt = GeoTransform::north_up(origin_x, args.resolution, origin_y, -args.resolution);

    let writer = GeoTiffWriter::new(w as u32, h as u32, 1)
        .layout(WriteLayout::Tiled {
            tile_width: 256,
            tile_height: 256,
        })
        .compression(Compression::Deflate)
        .sample_format(SampleFormat::IeeeFloat)
        .geo_transform(gt)
        .epsg(epsg)
        .no_data(args.nodata)
        .software(format!("laz2geotiff {}", env!("CARGO_PKG_VERSION")));

    // Write to a temp file first so a crash never leaves a half-written tile
    // that `resume` would wrongly skip.
    let tmp = path.with_extension("tif.tmp");
    writer
        .write_f32(&tmp, &data)
        .map_err(|e| anyhow::anyhow!("wbgeotiff write failed: {e}"))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
