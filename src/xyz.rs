//! `xyz`: a DTM straight on the Web Mercator XYZ pixel grid, as Int32 height
//! steps, for terrain-tiles.
//!
//! Points are reprojected with a PROJ pipeline (through a bilinear grid of
//! exact transforms), triangulated per output tile plus a margin, and each
//! pixel inside a kept triangle is interpolated by natural neighbours (or
//! linearly within its triangle).
//!
//! Work goes in blocks (tiles of `--block-zoom`): the points of a block are
//! read once and held in RAM, then its tiles render in parallel into one
//! GeoTIFF per block. A block file exists only once complete, so a re-run
//! resumes; `blocks.done` also records blocks that turned out empty.
//!
//! A flight strip is far longer than a block, so the first read of a file
//! records each LAZ chunk's EPSG:3857 box of kept points (`chunks.sqlite`);
//! later blocks decode only the chunks that reach them. Chunks decode in
//! parallel, not just files.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use gdal::raster::{Buffer, RasterCreationOptions};
use gdal::spatial_ref::SpatialRef;
use gdal::DriverManager;
use laz::{DecompressionSelection, LasZipDecompressor, LazVlr};
use proj::Proj;
use rayon::prelude::*;
use rusqlite::Connection;
use spade::{DelaunayTriangulation, HasPosition, HierarchyHintGenerator, Point2, Triangulation};

const HALF_WORLD: f64 = 20037508.342789244;
const NODATA: i32 = i32::MIN;

#[derive(Parser)]
pub struct XyzArgs {
    /// Sqlite bbox index built by `index` (source CRS).
    #[arg(long)]
    index: PathBuf,
    /// File holding the PROJ pipeline from the source CRS to EPSG:3857.
    #[arg(long)]
    ct: PathBuf,
    /// Output directory: one GeoTIFF per block, plus `blocks.done`.
    #[arg(long)]
    out_dir: PathBuf,
    /// Output zoom: pixels are this zoom's XYZ pixels.
    #[arg(long, default_value_t = 20)]
    zoom: u8,
    /// Zoom of the tiles triangulated one at a time (RAM per thread).
    #[arg(long, default_value_t = 17)]
    tile_zoom: u8,
    /// Zoom of the blocks whose points are held in RAM at once.
    #[arg(long, default_value_t = 13)]
    block_zoom: u8,
    /// Extra points around each tile, EPSG:3857 metres: past `--gap-cap`, so
    /// neighbouring tiles see the same triangles up to a gap's size.
    #[arg(long, default_value_t = 40.0)]
    margin: f64,
    /// A triangle is a gap when its longest edge passes this many times the
    /// local median; 0 leaves only `--gap-cap`.
    #[arg(long, default_value_t = 4.0)]
    gap_factor: f64,
    /// Longest edge (EPSG:3857 metres) past which a triangle is always a gap.
    #[arg(long, default_value_t = 30.0)]
    gap_cap: f64,
    /// Triangles with a longer edge (EPSG:3857 metres) stay nodata; 0 keeps
    /// all, leaving the outline to a cutline.
    #[arg(long, default_value_t = 0.0)]
    max_edge: f64,
    /// How a pixel's height comes from the surrounding points.
    #[arg(long, value_enum, default_value_t = Interpolation::Natural)]
    interpolation: Interpolation,
    /// LAS classes used.
    #[arg(long, value_delimiter = ',', default_value = "2")]
    classes: Vec<u8>,
    /// Height step of the Int32 output, metres.
    #[arg(long, default_value_t = 0.002)]
    step: f64,
    /// Spacing of the transform grid, source CRS units.
    #[arg(long, default_value_t = 100.0)]
    grid: f64,
    /// ZSTD level of the block files.
    #[arg(long, default_value_t = 9)]
    zstd_level: u8,
    #[arg(long, default_value_t = crate::default_jobs())]
    jobs: usize,
    /// Threads reading the next block's points while one renders.
    #[arg(long, default_value_t = 4)]
    read_jobs: usize,
    /// Memory (GB) a tile must leave available before it may start.
    #[arg(long, default_value_t = 4.0)]
    mem_reserve: f64,
    /// Limit "west,south,east,north", EPSG:3857 metres.
    #[arg(long)]
    bbox: Option<String>,
}

#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum Interpolation {
    /// Sibson natural neighbours: smooth across triangle edges.
    Natural,
    /// Barycentric within each triangle: faster, faceted.
    Linear,
}

/// A point in EPSG:3857 millimetres from `Ctx::origin`, so every block sees
/// the same coordinates for it.
#[derive(Clone, Copy)]
struct Pt {
    x: i32,
    y: i32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Vtx {
    p: Point2<f64>,
    z: f64,
}

impl HasPosition for Vtx {
    type Scalar = f64;
    fn position(&self) -> Point2<f64> {
        self.p
    }
}

/// Bilinear grid of exact source -> EPSG:3857 transforms.
struct Grid {
    x0: f64,
    y0: f64,
    step: f64,
    nx: usize,
    ny: usize,
    xy: Vec<[f64; 2]>,
}

impl Grid {
    fn build(ct: &str, ext: [f64; 4], step: f64) -> Result<Self> {
        let x0 = (ext[0] / step).floor() * step - step;
        let y0 = (ext[1] / step).floor() * step - step;
        let nx = ((ext[2] - x0) / step).ceil() as usize + 2;
        let ny = ((ext[3] - y0) / step).ceil() as usize + 2;
        Proj::new(ct).context("bad --ct pipeline")?;
        let rows: Vec<Vec<[f64; 2]>> = (0..ny)
            .into_par_iter()
            .map_init(
                || Proj::new(ct).unwrap(),
                |p, j| {
                    (0..nx)
                        .map(|i| {
                            let (x, y) = p
                                .convert((x0 + i as f64 * step, y0 + j as f64 * step))
                                .expect("transform failed");
                            [x, y]
                        })
                        .collect()
                },
            )
            .collect();
        Ok(Self { x0, y0, step, nx, ny, xy: rows.concat() })
    }

    fn apply(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let fx = (x - self.x0) / self.step;
        let fy = (y - self.y0) / self.step;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (i, j) = (fx as usize, fy as usize);
        if i + 1 >= self.nx || j + 1 >= self.ny {
            return None;
        }
        let (tx, ty) = (fx - i as f64, fy - j as f64);
        let k = j * self.nx + i;
        let (a, b, c, d) = (self.xy[k], self.xy[k + 1], self.xy[k + self.nx], self.xy[k + self.nx + 1]);
        let lerp = |n: usize| {
            (a[n] * (1.0 - tx) + b[n] * tx) * (1.0 - ty) + (c[n] * (1.0 - tx) + d[n] * tx) * ty
        };
        Some((lerp(0), lerp(1)))
    }
}

/// Per file, each chunk's EPSG:3857 box of kept points (`None`: it has none),
/// cached in sqlite across runs. Keyed by the kept classes too.
struct ChunkBoxes {
    conn: Mutex<Connection>,
    classes: String,
    map: Mutex<HashMap<String, Vec<Option<[f64; 4]>>>>,
}

impl ChunkBoxes {
    fn open(path: &std::path::Path, classes: &[u8]) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunk_boxes (file TEXT, classes TEXT, boxes BLOB, PRIMARY KEY (file, classes))",
        )?;
        let classes = classes.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
        let mut map = HashMap::new();
        {
            let mut stmt = conn.prepare("SELECT file, boxes FROM chunk_boxes WHERE classes = ?1")?;
            let rows = stmt.query_map([&classes], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
            for row in rows {
                let (file, blob) = row?;
                let boxes = blob
                    .chunks_exact(32)
                    .map(|b| {
                        let v: [f64; 4] = std::array::from_fn(|i| f64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()));
                        (!v[0].is_nan()).then_some(v)
                    })
                    .collect();
                map.insert(file, boxes);
            }
        }
        Ok(Self { conn: Mutex::new(conn), classes, map: Mutex::new(map) })
    }

    fn get(&self, file: &str) -> Option<Vec<Option<[f64; 4]>>> {
        self.map.lock().unwrap().get(file).cloned()
    }

    fn put(&self, file: &str, boxes: Vec<Option<[f64; 4]>>) -> Result<()> {
        let blob: Vec<u8> = boxes
            .iter()
            .flat_map(|b| b.unwrap_or([f64::NAN; 4]).into_iter().flat_map(f64::to_le_bytes))
            .collect();
        self.conn.lock().unwrap().execute(
            "INSERT OR REPLACE INTO chunk_boxes (file, classes, boxes) VALUES (?1, ?2, ?3)",
            rusqlite::params![file, self.classes, blob],
        )?;
        self.map.lock().unwrap().insert(file.to_string(), boxes);
        Ok(())
    }
}

/// A LAZ file opened at its point data, with what decoding it needs.
struct LazHeader {
    h: las::raw::Header,
    vlr: LazVlr,
    points: u64,
    chunk: u64,
    /// Point formats 0-5.
    legacy: bool,
}

impl LazHeader {
    fn chunks(&self) -> usize {
        self.points.div_ceil(self.chunk) as usize
    }
}

/// A triangle is a gap in the points when its longest edge is over
/// `--gap-factor` times the median in its `GAP_CELL` cell, within
/// `GAP_FLOOR..--gap-cap` (EPSG:3857 metres): natural neighbours would bulge
/// into a dome there, so it is left empty for the membrane fill. Relative, so
/// dense and sparse data both keep their ordinary triangles. Tiles agree on a
/// gap only while `--margin` covers it.
const GAP_CELL: f64 = 32.0;
const GAP_FLOOR: f64 = 2.0;

/// Chunks decoded by one task, each file split into runs of this many.
const CHUNKS_PER_TASK: usize = 8;

struct LazFile {
    path: String,
    /// EPSG:3857 w, s, e, n.
    bbox: [f64; 4],
}

struct Ctx<'a> {
    args: &'a XyzArgs,
    grid: Grid,
    files: Vec<LazFile>,
    keep: [bool; 256],
    chunks: ChunkBoxes,
    /// EPSG:3857 metres of `Pt` (0, 0).
    origin: (f64, f64),
    px: f64,
    gate: Gate,
}

/// Rough peak bytes per point of one tile's triangulation and lookups.
const BYTES_PER_POINT: u64 = 250;

/// How long a started tile's memory counts as not yet allocated: its
/// triangulation is built within this, so `MemAvailable` lags until then.
const RAMP: std::time::Duration = std::time::Duration::from_secs(10);

/// Lets a tile start only while `MemAvailable` leaves the reserve after it and
/// after tiles started too recently to show there, and under the job limit.
/// `<out-dir>/limits` ("<jobs> <reserve GB>") overrides both, re-read per tile.
struct Gate {
    limits: PathBuf,
    jobs: usize,
    reserve: f64,
    state: Mutex<GateState>,
    cv: std::sync::Condvar,
}

struct GateState {
    running: usize,
    recent: Vec<(Instant, u64)>,
}

struct Permit<'a>(&'a Gate);

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().running -= 1;
        self.0.cv.notify_all();
    }
}

impl Gate {
    fn new(args: &XyzArgs) -> Self {
        Gate {
            limits: args.out_dir.join("limits"),
            jobs: args.jobs,
            reserve: args.mem_reserve,
            state: Mutex::new(GateState { running: 0, recent: Vec::new() }),
            cv: std::sync::Condvar::new(),
        }
    }

    fn limits(&self) -> (usize, u64) {
        let parsed = std::fs::read_to_string(&self.limits).ok().and_then(|s| {
            let mut it = s.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next().map_or(Some(self.reserve), |v| v.parse().ok())?))
        });
        let (jobs, reserve) = parsed.unwrap_or((self.jobs, self.reserve));
        (jobs, (reserve * 1e9) as u64)
    }

    fn admit(&self, points: usize) -> Permit<'_> {
        let need = points as u64 * BYTES_PER_POINT;
        let mut st = self.state.lock().unwrap();
        loop {
            let (jobs, reserve) = self.limits();
            st.recent.retain(|(t, _)| t.elapsed() < RAMP);
            let pending: u64 = st.recent.iter().map(|r| r.1).sum();
            let free = mem_available().saturating_sub(pending);
            // The first tile always runs, or a too-big one would wait forever.
            if st.running == 0 || (st.running < jobs && free >= need + reserve) {
                st.running += 1;
                st.recent.push((Instant::now(), need));
                return Permit(self);
            }
            st = self.cv.wait_timeout(st, std::time::Duration::from_millis(500)).unwrap().0;
        }
    }
}

/// `MemAvailable` from /proc/meminfo, bytes; 0 if unreadable.
fn mem_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("MemAvailable:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

pub fn run(args: XyzArgs) -> Result<()> {
    ensure!(args.block_zoom <= args.tile_zoom && args.tile_zoom <= args.zoom);
    std::fs::create_dir_all(&args.out_dir)?;
    rayon::ThreadPoolBuilder::new().num_threads(args.jobs).build_global().ok();
    gdal::config::set_config_option("GDAL_CACHEMAX", "2048")?;

    let ct = std::fs::read_to_string(&args.ct)?.trim().to_string();

    let rows: Vec<(f64, f64, f64, f64, String)> = {
        let conn = Connection::open_with_flags(&args.index, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut stmt = conn.prepare("SELECT min_x, min_y, max_x, max_y, file FROM laz_index")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    ensure!(!rows.is_empty(), "empty index");

    let mut ext = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for r in &rows {
        ext = [ext[0].min(r.0), ext[1].min(r.1), ext[2].max(r.2), ext[3].max(r.3)];
    }
    let t = Instant::now();
    let grid = Grid::build(&ct, ext, args.grid)?;
    eprintln!("transform grid {}x{} in {:.1?}", grid.nx, grid.ny, t.elapsed());

    // A file's EPSG:3857 bbox from a 9x9 sample of its source bbox.
    let files: Vec<LazFile> = rows
        .into_iter()
        .map(|(x0, y0, x1, y1, path)| {
            let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = x0 + (x1 - x0) * i as f64 / 8.0;
                    let y = y0 + (y1 - y0) * j as f64 / 8.0;
                    let (mx, my) = grid.apply(x, y).expect("file outside the transform grid");
                    b = [b[0].min(mx), b[1].min(my), b[2].max(mx), b[3].max(my)];
                }
            }
            // curvature between samples
            LazFile { path, bbox: [b[0] - 1.0, b[1] - 1.0, b[2] + 1.0, b[3] + 1.0] }
        })
        .collect();

    let mut keep = [false; 256];
    for &c in &args.classes {
        keep[c as usize] = true;
    }

    let mut data_ext = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for f in &files {
        let b = f.bbox;
        data_ext = [data_ext[0].min(b[0]), data_ext[1].min(b[1]), data_ext[2].max(b[2]), data_ext[3].max(b[3])];
    }

    let ctx = Ctx {
        args: &args,
        grid,
        files,
        keep,
        chunks: ChunkBoxes::open(&args.out_dir.join("chunks.sqlite"), &args.classes)?,
        origin: (data_ext[0].floor() - 1000.0, data_ext[1].floor() - 1000.0),
        px: 2.0 * HALF_WORLD / 256.0 / (1u64 << args.zoom) as f64,
        gate: Gate::new(&args),
    };

    let lim = match &args.bbox {
        Some(b) => {
            let v: Vec<f64> = b.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
            ensure!(v.len() == 4, "--bbox needs west,south,east,north");
            [v[0].max(data_ext[0]), v[1].max(data_ext[1]), v[2].min(data_ext[2]), v[3].min(data_ext[3])]
        }
        None => data_ext,
    };

    let bsz = 2.0 * HALF_WORLD / (1u64 << args.block_zoom) as f64;
    let bx0 = ((lim[0] + HALF_WORLD) / bsz).floor() as i64;
    let bx1 = ((lim[2] + HALF_WORLD) / bsz).ceil() as i64 - 1;
    let by0 = ((HALF_WORLD - lim[3]) / bsz).floor() as i64;
    let by1 = ((HALF_WORLD - lim[1]) / bsz).ceil() as i64 - 1;

    let done_path = args.out_dir.join("blocks.done");
    let done: std::collections::HashSet<String> = std::fs::read_to_string(&done_path)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect();
    let mut done_log = OpenOptions::new().create(true).append(true).open(&done_path)?;

    // Serpentine, so consecutive blocks share files still in the page cache.
    let mut blocks = Vec::new();
    for by in by0..=by1 {
        let row: Vec<i64> = if (by - by0) % 2 == 0 {
            (bx0..=bx1).collect()
        } else {
            (bx0..=bx1).rev().collect()
        };
        for bx in row {
            let key = block_name(args.block_zoom, bx, by);
            if !done.contains(&key) && !ctx.block_files(bx, by, 0.0).is_empty() {
                blocks.push((bx, by));
            }
        }
    }

    eprintln!(
        "{} blocks to do (z{} {}..={} x {}..={}), px {} m, {} threads",
        blocks.len(),
        args.block_zoom,
        bx0,
        bx1,
        by0,
        by1,
        ctx.px,
        args.jobs
    );

    // The next block's points are read on a pool of their own while this block
    // renders: reading waits on the disk, and holds one block's points at most.
    let read_pool = rayon::ThreadPoolBuilder::new().num_threads(args.read_jobs).build()?;
    let pool = &read_pool;
    let ctx = &ctx;
    let t_all = Instant::now();
    std::thread::scope(|s| -> Result<()> {
        let mut next = blocks.first().map(|&(bx, by)| s.spawn(move || pool.install(|| ctx.read_block(bx, by))));
        for (i, &(bx, by)) in blocks.iter().enumerate() {
            let t = Instant::now();
            let (pts, stats) = next.take().expect("a read per block").join().expect("read thread panicked");
            let wait_s = t.elapsed().as_secs_f64();
            if let Some(&(nx, ny)) = blocks.get(i + 1) {
                next = Some(s.spawn(move || pool.install(|| ctx.read_block(nx, ny))));
            }
            ctx.render_block(bx, by, pts).with_context(|| format!("block {bx} {by}"))?;
            writeln!(done_log, "{}", block_name(args.block_zoom, bx, by))?;
            done_log.flush()?;
            let el = t_all.elapsed().as_secs_f64();
            eprintln!(
                "{} block {}/{} {bx} {by}: {} files {:.1} GB, {} pts, read {:.0}s (waited {:.0}s), render {:.0}s; {:.1} h elapsed, ~{:.1} h left",
                chrono_now(),
                i + 1,
                blocks.len(),
                stats.files,
                stats.bytes as f64 / 1e9,
                stats.points,
                stats.read_s,
                wait_s,
                t.elapsed().as_secs_f64() - wait_s,
                el / 3600.0,
                el / (i + 1) as f64 * (blocks.len() - i - 1) as f64 / 3600.0,
            );
        }
        Ok(())
    })?;
    eprintln!("done");
    Ok(())
}

fn block_name(z: u8, x: i64, y: i64) -> String {
    format!("{z}_{x}_{y}")
}

fn chrono_now() -> String {
    let out = std::process::Command::new("date").arg("-Is").output();
    out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

struct BlockStats {
    files: usize,
    bytes: u64,
    points: usize,
    read_s: f64,
}

impl Ctx<'_> {
    fn tile_bounds(&self, z: u8, x: i64, y: i64) -> [f64; 4] {
        let s = 2.0 * HALF_WORLD / (1u64 << z) as f64;
        [-HALF_WORLD + x as f64 * s, HALF_WORLD - (y + 1) as f64 * s, -HALF_WORLD + (x + 1) as f64 * s, HALF_WORLD - y as f64 * s]
    }

    fn block_files(&self, bx: i64, by: i64, m: f64) -> Vec<&LazFile> {
        let b = self.tile_bounds(self.args.block_zoom, bx, by);
        self.files
            .iter()
            .filter(|f| f.bbox[2] >= b[0] - m && f.bbox[0] <= b[2] + m && f.bbox[3] >= b[1] - m && f.bbox[1] <= b[3] + m)
            .collect()
    }

    fn to_mm(&self, x: f64, y: f64) -> (i32, i32) {
        (((x - self.origin.0) * 1000.0).round() as i32, ((y - self.origin.1) * 1000.0).round() as i32)
    }

    /// The points of a block and its margin.
    fn read_block(&self, bx: i64, by: i64) -> (Vec<Pt>, BlockStats) {
        let m = self.args.margin;
        let b = self.tile_bounds(self.args.block_zoom, bx, by);
        let files = self.block_files(bx, by, m);
        let bytes = files.iter().map(|f| std::fs::metadata(&f.path).map(|m| m.len()).unwrap_or(0)).sum();

        let t = Instant::now();
        let q = [b[0] - m, b[1] - m, b[2] + m, b[3] + m];
        let pts = Mutex::new(Vec::<Pt>::new());
        self.read_block_points(&files, q, &pts);
        let pts = pts.into_inner().unwrap();
        let stats = BlockStats { files: files.len(), bytes, points: pts.len(), read_s: t.elapsed().as_secs_f64() };
        (pts, stats)
    }

    fn render_block(&self, bx: i64, by: i64, mut pts: Vec<Pt>) -> Result<()> {
        let a = self.args;
        let out = a.out_dir.join(format!("{}.tif", block_name(a.block_zoom, bx, by)));
        let tmp = out.with_extension("tif.tmp");
        let m = a.margin;
        let b = self.tile_bounds(a.block_zoom, bx, by);
        if pts.is_empty() {
            return Ok(());
        }

        // Bucket by tile; margin points fall in the ring of rows/cols -1 and n.
        let n = 1i64 << (a.tile_zoom - a.block_zoom);
        let tsz = 2.0 * HALF_WORLD / (1u64 << a.tile_zoom) as f64;
        ensure!(m <= tsz, "margin exceeds a tile");
        let (bx0mm, by1mm) = self.to_mm(b[0], b[3]);
        let tmm = tsz * 1000.0;
        let key = |p: &Pt| -> u32 {
            let c = (((p.x - bx0mm) as f64 / tmm).floor() as i64).clamp(-1, n);
            let r = (((by1mm - p.y) as f64 / tmm).floor() as i64).clamp(-1, n);
            ((r + 1) * (n + 2) + (c + 1)) as u32
        };
        pts.par_sort_unstable_by_key(key);
        let nk = ((n + 2) * (n + 2)) as usize;
        let mut starts = vec![0usize; nk + 1];
        {
            let mut k = 0usize;
            for (i, p) in pts.iter().enumerate() {
                let pk = key(p) as usize;
                while k <= pk {
                    starts[k] = i;
                    k += 1;
                }
            }
            while k <= nk {
                starts[k] = pts.len();
                k += 1;
            }
        }

        let side = 256usize << (a.zoom - a.block_zoom);
        let tside = 256usize << (a.zoom - a.tile_zoom);

        let ds = {
            let drv = DriverManager::get_driver_by_name("GTiff")?;
            let level = format!("ZSTD_LEVEL={}", a.zstd_level);
            let opts = RasterCreationOptions::from_iter([
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "COMPRESS=ZSTD",
                level.as_str(),
                "PREDICTOR=2",
                "SPARSE_OK=TRUE",
                "BIGTIFF=IF_SAFER",
                "NUM_THREADS=4",
            ]);
            let mut ds = drv.create_with_band_type_with_options::<i32, _>(&tmp, side, side, 1, &opts)?;
            ds.set_geo_transform(&[b[0], self.px, 0.0, b[3], 0.0, -self.px])?;
            ds.set_spatial_ref(&SpatialRef::from_epsg(3857)?)?;
            let mut band = ds.rasterband(1)?;
            band.set_no_data_value(Some(NODATA as f64))?;
            band.set_scale(a.step)?;
            band.set_offset(0.0)?;
            Mutex::new(ds)
        };

        let tiles: Vec<(i64, i64)> = (0..n).flat_map(|r| (0..n).map(move |c| (c, r))).collect();
        tiles.par_iter().try_for_each(|&(c, r)| -> Result<()> {
            let mut sel = Vec::new();
            let tb = [b[0] + c as f64 * tsz, b[3] - (r + 1) as f64 * tsz, b[0] + (c + 1) as f64 * tsz, b[3] - r as f64 * tsz];
            let (qx0, qy0) = self.to_mm(tb[0] - m, tb[1] - m);
            let (qx1, qy1) = self.to_mm(tb[2] + m, tb[3] + m);
            let (ix0, iy0) = self.to_mm(tb[0], tb[1]);
            let (ix1, iy1) = self.to_mm(tb[2], tb[3]);
            let mut inside = false;
            for rr in r..=r + 2 {
                for cc in c..=c + 2 {
                    let k = (rr * (n + 2) + cc) as usize;
                    for p in &pts[starts[k]..starts[k + 1]] {
                        if p.x >= qx0 && p.x <= qx1 && p.y >= qy0 && p.y <= qy1 {
                            inside |= p.x >= ix0 && p.x < ix1 && p.y >= iy0 && p.y < iy1;
                            sel.push(*p);
                        }
                    }
                }
            }
            if !inside {
                return Ok(());
            }
            let _permit = self.gate.admit(sel.len());
            let Some(data) = self.render_tile(sel, tb, tside)? else {
                return Ok(());
            };
            let mut buf = Buffer::new((tside, tside), data);
            let ds = ds.lock().unwrap();
            ds.rasterband(1)?
                .write(((c as usize * tside) as isize, (r as usize * tside) as isize), (tside, tside), &mut buf)?;
            Ok(())
        })?;
        drop(pts);

        let mut ds = ds.into_inner().unwrap();
        ds.flush_cache()?;
        drop(ds);
        std::fs::rename(&tmp, &out)?;
        Ok(())
    }

    /// TIN heights in steps on the tile's pixel grid; None if all nodata.
    fn render_tile(&self, mut sel: Vec<Pt>, tb: [f64; 4], side: usize) -> Result<Option<Vec<i32>>> {
        // Deterministic duplicate choice, so neighbouring tiles agree.
        sel.sort_unstable_by(|a, b| (a.x, a.y).cmp(&(b.x, b.y)).then(a.z.total_cmp(&b.z)));
        sel.dedup_by(|a, b| a.x == b.x && a.y == b.y);
        if sel.len() < 3 {
            return Ok(None);
        }

        let px = self.px;
        let natural = self.args.interpolation == Interpolation::Natural;
        // Tile-relative metres keep the predicates well conditioned.
        let (ox, oy) = (tb[0] - self.origin.0, tb[3] - self.origin.1);
        let vtx: Vec<Vtx> = sel
            .iter()
            .map(|p| Vtx { p: Point2::new(p.x as f64 / 1000.0 - ox, p.y as f64 / 1000.0 - oy), z: p.z as f64 })
            .collect();
        drop(sel);
        // Lookups never move the default hint, so each pixel's natural-neighbour
        // search would walk from one vertex; the hierarchy starts it nearby.
        let dt = DelaunayTriangulation::<Vtx, (), (), (), HierarchyHintGenerator<f64>>::bulk_load(vtx)
            .map_err(|e| anyhow::anyhow!("triangulation: {e:?}"))?;

        let inv_step = 1.0 / self.args.step;
        let max_e2 = if self.args.max_edge > 0.0 { self.args.max_edge * self.args.max_edge } else { f64::INFINITY };
        let longest = |a: &Vtx, b: &Vtx, c: &Vtx| {
            let d2 = |p: Point2<f64>, q: Point2<f64>| (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y);
            d2(a.p, b.p).max(d2(b.p, c.p)).max(d2(c.p, a.p)).sqrt()
        };

        // Per `GAP_CELL` cell on a grid fixed in EPSG:3857, so neighbouring
        // tiles see the same cells: the median longest edge of the triangles
        // centred in it.
        let m = self.args.margin;
        let (cx0, cy0) = (((tb[0] - m) / GAP_CELL).floor(), ((tb[1] - m) / GAP_CELL).floor());
        let ncx = ((tb[2] + m) / GAP_CELL).floor() as usize - cx0 as usize + 1;
        let ncy = ((tb[3] + m) / GAP_CELL).floor() as usize - cy0 as usize + 1;
        let cell = |a: &Vtx, b: &Vtx, c: &Vtx| -> Option<usize> {
            let x = ((a.p.x + b.p.x + c.p.x) / 3.0 + tb[0]) / GAP_CELL - cx0;
            let y = ((a.p.y + b.p.y + c.p.y) / 3.0 + tb[3]) / GAP_CELL - cy0;
            (x >= 0.0 && y >= 0.0 && (x as usize) < ncx && (y as usize) < ncy).then(|| y as usize * ncx + x as usize)
        };
        let gap_at: Vec<f64> = if natural {
            let mut lens: Vec<Vec<f32>> = vec![Vec::new(); ncx * ncy];
            for f in dt.inner_faces() {
                let [va, vb, vc] = f.vertices();
                let (a, b, c) = (va.data(), vb.data(), vc.data());
                if let Some(i) = cell(a, b, c) {
                    lens[i].push(longest(a, b, c) as f32);
                }
            }
            let (factor, cap) = (self.args.gap_factor, self.args.gap_cap);
            lens.into_iter()
                .map(|mut l| {
                    if l.is_empty() || factor <= 0.0 {
                        return cap;
                    }
                    let mid = l.len() / 2;
                    let median = *l.select_nth_unstable_by(mid, f32::total_cmp).1 as f64;
                    (factor * median).clamp(GAP_FLOOR.min(cap), cap)
                })
                .collect()
        } else {
            Vec::new()
        };

        let nn = dt.natural_neighbor();
        let mut out = vec![NODATA; side * side];
        let mut any = false;
        let last = side as f64 - 1.0;

        // Pixel (c, r) centre: x = (c + .5) px, y = -(r + .5) px.
        for f in dt.inner_faces() {
            let [va, vb, vc] = f.vertices();
            let (a, b, c) = (va.data(), vb.data(), vc.data());
            let (ax, ay, bx, by, cx, cy) = (a.p.x, a.p.y, b.p.x, b.p.y, c.p.x, c.p.y);
            let e = longest(a, b, c);
            if e * e > max_e2 {
                continue;
            }
            if natural && e > cell(a, b, c).map_or(self.args.gap_cap, |i| gap_at[i]) {
                continue;
            }
            let c0 = (ax.min(bx).min(cx) / px - 0.5).ceil().max(0.0);
            let c1 = (ax.max(bx).max(cx) / px - 0.5).floor().min(last);
            let r0 = (-(ay.max(by).max(cy)) / px - 0.5).ceil().max(0.0);
            let r1 = (-(ay.min(by).min(cy)) / px - 0.5).floor().min(last);
            if c0 > c1 || r0 > r1 {
                continue;
            }
            let det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let eps = -1e-9;
            for r in r0 as usize..=r1 as usize {
                let y = -(r as f64 + 0.5) * px;
                let row = r * side;
                for col in c0 as usize..=c1 as usize {
                    let x = (col as f64 + 0.5) * px;
                    let l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) * inv;
                    let l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) * inv;
                    let l3 = 1.0 - l1 - l2;
                    if l1 >= eps && l2 >= eps && l3 >= eps {
                        let z = if natural {
                            match nn.interpolate(|v| v.data().z, Point2::new(x, y)) {
                                Some(z) => z,
                                None => continue,
                            }
                        } else {
                            l1 * a.z + l2 * b.z + l3 * c.z
                        };
                        out[row + col] = (z * inv_step).round() as i32;
                        any = true;
                    }
                }
            }
        }
        Ok(any.then_some(out))
    }

    /// Appends the kept points of `files` inside `q` (EPSG:3857 w, s, e, n),
    /// decoding only the chunks known to reach `q`; a file read for the first
    /// time is decoded whole and its chunk boxes recorded.
    fn read_block_points(&self, files: &[&LazFile], q: [f64; 4], out: &Mutex<Vec<Pt>>) {
        let meets = |b: &[f64; 4]| b[2] >= q[0] && b[0] <= q[2] && b[3] >= q[1] && b[1] <= q[3];

        // (file, chunks, whether their boxes are being recorded)
        let mut tasks: Vec<(usize, Vec<usize>, bool)> = Vec::new();
        let mut unknown: Vec<(usize, usize)> = Vec::new();
        for (i, f) in files.iter().enumerate() {
            let (want, record): (Vec<usize>, bool) = match self.chunks.get(&f.path) {
                Some(boxes) => ((0..boxes.len()).filter(|&c| boxes[c].as_ref().is_some_and(meets)).collect(), false),
                None => match open_laz(&f.path) {
                    Ok((_, lh)) => {
                        unknown.push((i, lh.chunks()));
                        ((0..lh.chunks()).collect(), true)
                    }
                    Err(e) => {
                        eprintln!("WARN {}: {e:#}", f.path);
                        continue;
                    }
                },
            };
            for run in want.chunks(CHUNKS_PER_TASK) {
                tasks.push((i, run.to_vec(), record));
            }
        }

        let found: Mutex<HashMap<usize, Vec<(usize, Option<[f64; 4]>)>>> = Mutex::default();
        let failed: Mutex<std::collections::HashSet<usize>> = Mutex::default();
        tasks.par_iter().for_each(|(i, run, record)| match self.read_chunks(&files[*i].path, run, q, out) {
            Ok(boxes) => {
                if *record {
                    found.lock().unwrap().entry(*i).or_default().extend(boxes);
                }
            }
            Err(e) => {
                eprintln!("WARN {}: {e:#}", files[*i].path);
                failed.lock().unwrap().insert(*i);
            }
        });

        let (mut found, failed) = (found.into_inner().unwrap(), failed.into_inner().unwrap());
        for (i, n) in unknown {
            if failed.contains(&i) {
                continue;
            }
            let mut boxes = vec![None; n];
            for (c, b) in found.remove(&i).unwrap_or_default() {
                boxes[c] = b;
            }
            if let Err(e) = self.chunks.put(&files[i].path, boxes) {
                eprintln!("WARN {}: chunk boxes not saved: {e:#}", files[i].path);
            }
        }
    }

    /// Appends the kept points inside `q` of the given chunks of `path`, and
    /// returns each chunk's box of kept points.
    fn read_chunks(&self, path: &str, run: &[usize], q: [f64; 4], out: &Mutex<Vec<Pt>>) -> Result<Vec<(usize, Option<[f64; 4]>)>> {
        let (f, lh) = open_laz(path)?;
        let h = &lh.h;
        let len = h.point_data_record_length as usize;
        // Only the layered formats 6+ can skip fields.
        let sel = if lh.legacy {
            DecompressionSelection::all()
        } else {
            DecompressionSelection(DecompressionSelection::Z | DecompressionSelection::CLASSIFICATION)
        };
        let mut dec = LasZipDecompressor::selective(f, lh.vlr.clone(), sel)?;
        // Formats 0-5 keep the class in byte 15's low 5 bits, 6+ in byte 16.
        let class = |rec: &[u8]| if lh.legacy { rec[15] & 0x1f } else { rec[16] };

        let mut buf = vec![0u8; lh.chunk as usize * len];
        let mut local = Vec::new();
        let mut boxes = Vec::with_capacity(run.len());
        let i32_at = |r: &[u8], o: usize| i32::from_le_bytes(r[o..o + 4].try_into().unwrap());
        for &c in run {
            let first = c as u64 * lh.chunk;
            let k = (lh.points - first).min(lh.chunk) as usize;
            dec.seek(first)?;
            dec.decompress_many(&mut buf[..k * len])?;
            let mut bx: Option<[f64; 4]> = None;
            for rec in buf[..k * len].chunks_exact(len) {
                if !self.keep[class(rec) as usize] {
                    continue;
                }
                let x = i32_at(rec, 0) as f64 * h.x_scale_factor + h.x_offset;
                let y = i32_at(rec, 4) as f64 * h.y_scale_factor + h.y_offset;
                let Some((mx, my)) = self.grid.apply(x, y) else { continue };
                bx = Some(match bx {
                    Some(b) => [b[0].min(mx), b[1].min(my), b[2].max(mx), b[3].max(my)],
                    None => [mx, my, mx, my],
                });
                if mx < q[0] || mx > q[2] || my < q[1] || my > q[3] {
                    continue;
                }
                let z = i32_at(rec, 8) as f64 * h.z_scale_factor + h.z_offset;
                let (ix, iy) = self.to_mm(mx, my);
                local.push(Pt { x: ix, y: iy, z: z as f32 });
            }
            boxes.push((c, bx));
        }
        out.lock().unwrap().append(&mut local);
        Ok(boxes)
    }
}

/// Opens a LAZ file with fixed-size chunks, positioned at its point data.
fn open_laz(path: &str) -> Result<(BufReader<File>, LazHeader)> {
    let mut f = BufReader::with_capacity(1 << 20, File::open(path)?);
    let h = las::raw::Header::read_from(&mut f)?;
    f.seek(SeekFrom::Start(h.header_size as u64))?;
    let mut vlr = None;
    for _ in 0..h.number_of_variable_length_records {
        let v = las::raw::Vlr::read_from(&mut f, false)?;
        if v.record_id == 22204 && v.user_id.starts_with(b"laszip encoded") {
            vlr = Some(LazVlr::from_buffer(&v.data)?);
        }
    }
    let Some(vlr) = vlr else { bail!("not a LAZ file") };
    let format = h.point_data_record_format & 0x3f;
    ensure!(format <= 10, "point format {format} unsupported");
    // A chunk's first point is then a multiple of the chunk size.
    ensure!(!vlr.uses_variable_size_chunks(), "variable-size chunks unsupported");
    let points = h.large_file.map(|l| l.number_of_point_records).unwrap_or(h.number_of_point_records as u64);
    let chunk = vlr.chunk_size() as u64;
    f.seek(SeekFrom::Start(h.offset_to_point_data as u64))?;
    Ok((f, LazHeader { h, vlr, points, chunk, legacy: format < 6 }))
}
