//! `fill`: fills the gaps an `xyz` render leaves (roofs, rivers, lakes) with
//! the membrane (Laplace) surface spanning their rims: smooth, and level where
//! the rim is level.
//!
//! A gap can be far larger than any window a block could solve alone, so the
//! whole mosaic is first filled at a coarse zoom in one piece. Each block is
//! then solved at full resolution with its rims from the render and, at the
//! edge of its window, the coarse field: neighbouring blocks share the same
//! field, so they agree where they meet.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use gdal::raster::{Buffer, RasterCreationOptions, ResampleAlg};
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DriverManager};
use rayon::prelude::*;

const HALF_WORLD: f64 = 20037508.342789244;
const NODATA: i32 = i32::MIN;

#[derive(Parser)]
pub struct FillArgs {
    /// VRT over the `xyz` block files.
    #[arg(long)]
    input: PathBuf,
    /// The `xyz` output directory holding the block files.
    #[arg(long)]
    blocks: PathBuf,
    /// Where the filled block files go, under the same names.
    #[arg(long)]
    out_dir: PathBuf,
    /// Zoom of the coarse fill of the whole mosaic.
    #[arg(long, default_value_t = 12)]
    coarse_zoom: u8,
    /// Pixels read around each block, whose outer edge takes the coarse field.
    #[arg(long, default_value_t = 512)]
    halo: usize,
    /// Blocks filled at once (each holds its window in RAM).
    #[arg(long, default_value_t = 8)]
    jobs: usize,
    /// A gap open to the edge of the data is filled this far from the data,
    /// EPSG:3857 metres; one the data encloses is filled whole.
    #[arg(long, default_value_t = 2000.0)]
    reach: f64,
    /// A VRT over a previous pass's filled blocks. It holds each window's edge
    /// instead of the coarse field, and starts the solve, so a gap spanning
    /// blocks settles on one surface; each pass narrows their disagreement.
    #[arg(long)]
    boundary: Option<PathBuf>,
}

pub fn run(args: FillArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out_dir)?;
    rayon::ThreadPoolBuilder::new().num_threads(args.jobs).build_global().ok();
    gdal::config::set_config_option("GDAL_CACHEMAX", "1024")?;

    let ds = Dataset::open(&args.input)?;
    let (w, h) = ds.raster_size();
    let gt = ds.geo_transform()?;
    let px = gt[1];
    let zoom = (2.0 * HALF_WORLD / 256.0 / px).log2().round() as u8;
    ensure!(args.coarse_zoom < zoom, "--coarse-zoom must be below the input's zoom {zoom}");
    let band = ds.rasterband(1)?;
    let step = band.scale().unwrap_or(1.0);
    let f = 1usize << (zoom - args.coarse_zoom);
    ensure!(w % f == 0 && h % f == 0, "input size {w}x{h} is not a multiple of {f}");
    drop(band);
    drop(ds);

    let coarse = Coarse::load_or_build(&args, (w, h), f, step)?;

    // Every block of the grid the files lie on, those without a file too: one
    // entirely inside a lake had no points, yet the coarse field reaches it.
    let files: Vec<PathBuf> = std::fs::read_dir(&args.blocks)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tif"))
        .collect();
    ensure!(!files.is_empty(), "no block files in {}", args.blocks.display());
    let bz: u8 = files[0].file_stem().unwrap().to_string_lossy().split('_').next().unwrap().parse()?;
    let bpx = 256usize << (zoom - bz);
    ensure!(w % bpx == 0 && h % bpx == 0, "input is not on the z{bz} block grid");
    let bsz = bpx as f64 * px;
    let (gx, gy) = (((gt[0] + HALF_WORLD) / bsz).round() as i64, ((HALF_WORLD - gt[3]) / bsz).round() as i64);
    let mut blocks = Vec::new();
    for j in 0..(h / bpx) as i64 {
        for i in 0..(w / bpx) as i64 {
            let name = format!("{bz}_{}_{}.tif", gx + i, gy + j);
            if args.out_dir.join(&name).exists() {
                continue;
            }
            let src = args.blocks.join(&name);
            let (c, r) = (i as usize * bpx, j as usize * bpx);
            if src.exists() || coarse.reaches(c, r, bpx) {
                blocks.push(Block { name, src: src.exists().then_some(src), col: c, row: r, side: bpx });
            }
        }
    }
    eprintln!("{} blocks to fill, zoom {zoom}, coarse zoom {} ({}x{})", blocks.len(), args.coarse_zoom, coarse.w, coarse.h);

    let t = Instant::now();
    let done = AtomicUsize::new(0);
    blocks.par_iter().try_for_each(|b| -> Result<()> {
        let filled = fill_block(&args, b, gt, step, &coarse).with_context(|| b.name.clone())?;
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!("{}/{} {}: {} px filled; {:.1} h elapsed", n, blocks.len(), b.name, filled, t.elapsed().as_secs_f64() / 3600.0);
        Ok(())
    })?;
    eprintln!("done");
    Ok(())
}

/// A block of the input's grid; `src` is its `xyz` file, if it has one.
struct Block {
    name: String,
    src: Option<PathBuf>,
    col: usize,
    row: usize,
    side: usize,
}

/// The mosaic at the coarse zoom with its gaps filled (`fillable_gaps`), in metres.
struct Coarse {
    v: Vec<f32>,
    w: usize,
    h: usize,
    /// Input pixels per coarse pixel.
    f: usize,
}

impl Coarse {
    fn load_or_build(args: &FillArgs, (w, h): (usize, usize), f: usize, step: f64) -> Result<Self> {
        let (cw, ch) = (w / f, h / f);
        let path = args.out_dir.join(format!("coarse_z{}.f32", args.coarse_zoom));
        if let Ok(bytes) = std::fs::read(&path) {
            if bytes.len() == cw * ch * 4 {
                eprintln!("coarse field from {}", path.display());
                let v = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                return Ok(Self { v, w: cw, h: ch, f });
            }
        }

        let t = Instant::now();
        const ROWS: usize = 16;
        let strips: Vec<Vec<f64>> = (0..ch.div_ceil(ROWS))
            .into_par_iter()
            .map(|s| -> Result<Vec<f64>> {
                let ds = Dataset::open(&args.input)?;
                let band = ds.rasterband(1)?;
                let rows = ROWS.min(ch - s * ROWS);
                let buf = band.read_as::<f64>(
                    (0, (s * ROWS * f) as isize),
                    (w, rows * f),
                    (cw, rows),
                    Some(ResampleAlg::Average),
                )?;
                Ok(buf.data().iter().map(|&x| if x < -2e9 { f64::NAN } else { x * step }).collect())
            })
            .collect::<Result<_>>()?;
        let mut v: Vec<f64> = strips.concat();
        eprintln!("coarse read {cw}x{ch} in {:.1?}", t.elapsed());

        let t = Instant::now();
        let coarse_px = 2.0 * HALF_WORLD / 256.0 / (1u64 << args.coarse_zoom) as f64;
        let gaps = fillable_gaps(&v, cw, ch, args.reach / coarse_px);
        membrane(&mut v, cw, ch, &gaps, true);
        eprintln!("coarse fill of {} px in {:.1?}", gaps.len(), t.elapsed());

        let v: Vec<f32> = v.into_iter().map(|x| x as f32).collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&path, bytes)?;
        Ok(Self { v, w: cw, h: ch, f })
    }

    /// Whether any coarse pixel over the input square at (col, row) has a value.
    fn reaches(&self, col: usize, row: usize, side: usize) -> bool {
        let (c0, r0, n) = (col / self.f, row / self.f, side.div_ceil(self.f));
        (r0..(r0 + n).min(self.h)).any(|r| (c0..(c0 + n).min(self.w)).any(|c| self.v[r * self.w + c].is_finite()))
    }

    /// Bilinear at the centre of input pixel (col, row), over the finite
    /// neighbours only; NaN where none is.
    fn at(&self, col: usize, row: usize) -> f64 {
        let x = (col as f64 + 0.5) / self.f as f64 - 0.5;
        let y = (row as f64 + 0.5) / self.f as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (mut sum, mut wsum) = (0.0, 0.0);
        for (dx, dy, wt) in [(0, 0, (1.0 - tx) * (1.0 - ty)), (1, 0, tx * (1.0 - ty)), (0, 1, (1.0 - tx) * ty), (1, 1, tx * ty)] {
            let (cx, cy) = (x0 as i64 + dx, y0 as i64 + dy);
            if cx < 0 || cy < 0 || cx >= self.w as i64 || cy >= self.h as i64 {
                continue;
            }
            let z = self.v[cy as usize * self.w + cx as usize] as f64;
            if z.is_finite() && wt > 0.0 {
                sum += z * wt;
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            sum / wsum
        } else {
            f64::NAN
        }
    }
}

/// Fills the gaps of one block and writes it to `--out-dir`; returns the
/// number of its pixels filled.
fn fill_block(args: &FillArgs, block: &Block, gt: [f64; 6], step: f64, coarse: &Coarse) -> Result<usize> {
    let out = args.out_dir.join(&block.name);
    let (bw, bh) = (block.side, block.side);
    let bgt = [gt[0] + block.col as f64 * gt[1], gt[1], 0.0, gt[3] - block.row as f64 * gt[1], 0.0, gt[5]];
    let core = match &block.src {
        Some(src) => {
            let bds = Dataset::open(src)?;
            ensure!(bds.raster_size() == (bw, bh), "unexpected block size");
            let core = bds.rasterband(1)?.read_as::<i32>((0, 0), (bw, bh), (bw, bh), None)?.into_shape_and_vec().1;
            if !core.contains(&NODATA) {
                link_or_copy(src, &out)?;
                return Ok(0);
            }
            core
        }
        None => vec![NODATA; bw * bh],
    };

    // The block's window with the halo.
    let (bx, by) = (block.col as i64, block.row as i64);
    let (iw, ih) = (coarse.w * coarse.f, coarse.h * coarse.f);
    let hal = args.halo as i64;
    let x0 = (bx - hal).max(0) as usize;
    let y0 = (by - hal).max(0) as usize;
    let x1 = ((bx + bw as i64 + hal) as usize).min(iw);
    let y1 = ((by + bh as i64 + hal) as usize).min(ih);
    let (ww, wh) = (x1 - x0, y1 - y0);

    let ds = Dataset::open(&args.input)?;
    let raw = ds.rasterband(1)?.read_as::<i32>((x0 as isize, y0 as isize), (ww, wh), (ww, wh), None)?;
    drop(ds);
    let mut v: Vec<f64> = raw.data().iter().map(|&z| if z == NODATA { f64::NAN } else { z as f64 * step }).collect();
    drop(raw);

    // The previous pass's fill, where one is given.
    let prev: Option<Vec<f64>> = match &args.boundary {
        Some(path) => {
            let ds = Dataset::open(path)?;
            let raw = ds.rasterband(1)?.read_as::<i32>((x0 as isize, y0 as isize), (ww, wh), (ww, wh), None)?;
            Some(raw.data().iter().map(|&z| if z == NODATA { f64::NAN } else { z as f64 * step }).collect())
        }
        None => None,
    };

    // A gap pixel is solved where the coarse field reaches it; on the window's
    // edge it takes the previous pass's value, else the coarse one, and holds it.
    let mut cells = Vec::new();
    for r in 0..wh {
        for c in 0..ww {
            let i = r * ww + c;
            if !v[i].is_nan() || coarse.at(x0 + c, y0 + r).is_nan() {
                continue;
            }
            let before = prev.as_ref().map_or(f64::NAN, |p| p[i]);
            if r > 0 && c > 0 && r + 1 < wh && c + 1 < ww {
                cells.push(i);
                v[i] = before;
            } else {
                v[i] = if before.is_nan() { coarse.at(x0 + c, y0 + r) } else { before };
            }
        }
    }
    if prev.is_some() {
        relax(&mut v, ww, wh, &cells, 20_000);
    } else {
        membrane(&mut v, ww, wh, &cells, false);
    }

    let (ox, oy) = ((bx as usize) - x0, (by as usize) - y0);
    let mut data = core;
    let mut filled = 0;
    for r in 0..bh {
        for c in 0..bw {
            let k = r * bw + c;
            if data[k] == NODATA {
                let z = v[(r + oy) * ww + c + ox];
                if z.is_finite() {
                    data[k] = (z / step).round() as i32;
                    filled += 1;
                }
            }
        }
    }

    let tmp = out.with_extension("tif.tmp");
    {
        let drv = DriverManager::get_driver_by_name("GTiff")?;
        let opts = RasterCreationOptions::from_iter([
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "COMPRESS=ZSTD",
            "ZSTD_LEVEL=9",
            "PREDICTOR=2",
            "SPARSE_OK=TRUE",
            "BIGTIFF=IF_SAFER",
            "NUM_THREADS=4",
        ]);
        let mut ds = drv.create_with_band_type_with_options::<i32, _>(&tmp, bw, bh, 1, &opts)?;
        ds.set_geo_transform(&bgt)?;
        ds.set_spatial_ref(&SpatialRef::from_epsg(3857)?)?;
        let mut band = ds.rasterband(1)?;
        band.set_no_data_value(Some(NODATA as f64))?;
        band.set_scale(step)?;
        band.set_offset(0.0)?;
        band.write((0, 0), (bw, bh), &mut Buffer::new((bw, bh), data))?;
    }
    std::fs::rename(&tmp, &out)?;
    Ok(filled)
}

fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    if std::fs::hard_link(from, to).is_err() {
        std::fs::copy(from, to)?;
    }
    Ok(())
}

/// The NaN pixels to fill: every one whose 4-connected region does not reach
/// the grid's edge, and of the rest those within `reach` pixels of data — a
/// river leaving across a border, a lake the border cuts — so what lies beyond
/// the data stays empty.
fn fillable_gaps(v: &[f64], w: usize, h: usize, reach: f64) -> Vec<usize> {
    let near = near_data(v, w, h, reach);
    let mut seen = vec![false; w * h];
    let mut gaps = Vec::new();
    let (mut stack, mut comp) = (Vec::new(), Vec::new());
    for s in 0..w * h {
        if seen[s] || !v[s].is_nan() {
            continue;
        }
        seen[s] = true;
        stack.push(s);
        comp.clear();
        let mut open = false;
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (r, c) = (i / w, i % w);
            open |= r == 0 || c == 0 || r + 1 == h || c + 1 == w;
            for j in neighbours(i, w, h) {
                if !seen[j] && v[j].is_nan() {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if open {
            gaps.extend(comp.iter().filter(|&&i| near[i]));
        } else {
            gaps.extend_from_slice(&comp);
        }
    }
    gaps
}

/// Whether each pixel lies within `reach` pixels of a finite one, by a
/// two-pass chamfer distance.
fn near_data(v: &[f64], w: usize, h: usize, reach: f64) -> Vec<bool> {
    const D: f64 = std::f64::consts::SQRT_2;
    let mut d: Vec<f64> = v.iter().map(|x| if x.is_nan() { f64::INFINITY } else { 0.0 }).collect();
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            let mut m = d[i];
            if c > 0 {
                m = m.min(d[i - 1] + 1.0);
            }
            if r > 0 {
                m = m.min(d[i - w] + 1.0);
                if c > 0 {
                    m = m.min(d[i - w - 1] + D);
                }
                if c + 1 < w {
                    m = m.min(d[i - w + 1] + D);
                }
            }
            d[i] = m;
        }
    }
    for r in (0..h).rev() {
        for c in (0..w).rev() {
            let i = r * w + c;
            let mut m = d[i];
            if c + 1 < w {
                m = m.min(d[i + 1] + 1.0);
            }
            if r + 1 < h {
                m = m.min(d[i + w] + 1.0);
                if c + 1 < w {
                    m = m.min(d[i + w + 1] + D);
                }
                if c > 0 {
                    m = m.min(d[i + w - 1] + D);
                }
            }
            d[i] = m;
        }
    }
    d.into_iter().map(|x| x <= reach).collect()
}

fn neighbours(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (r, c) = (i / w, i % w);
    [
        (r > 0).then(|| i - w),
        (r + 1 < h).then(|| i + w),
        (c > 0).then(|| i - 1),
        (c + 1 < w).then(|| i + 1),
    ]
    .into_iter()
    .flatten()
}

/// Fills `cells` (NaN in `v`) with the membrane over the finite pixels. Start
/// values come from a pyramid of the known heights, relaxed level by level on
/// the way down, so the final relaxation converges fast.
fn membrane(v: &mut [f64], w: usize, h: usize, cells: &[usize], log: bool) {
    if cells.is_empty() {
        return;
    }
    let mut levels = vec![(v.to_vec(), w, h)];
    while levels.last().map(|l| l.1.max(l.2) > 1).unwrap() {
        let (p, pw, ph) = levels.last().unwrap();
        let (p, pw, ph) = (p.as_slice(), *pw, *ph);
        let (tw, th) = (pw.div_ceil(2), ph.div_ceil(2));
        let mut q = vec![f64::NAN; tw * th];
        for r in 0..th {
            for c in 0..tw {
                let (mut sum, mut k) = (0.0, 0);
                for (rr, cc) in [(2 * r, 2 * c), (2 * r, 2 * c + 1), (2 * r + 1, 2 * c), (2 * r + 1, 2 * c + 1)] {
                    if rr < ph && cc < pw && p[rr * pw + cc].is_finite() {
                        sum += p[rr * pw + cc];
                        k += 1;
                    }
                }
                if k > 0 {
                    q[r * tw + c] = sum / k as f64;
                }
            }
        }
        levels.push((q, tw, th));
    }
    for l in (0..levels.len() - 1).rev() {
        let (upper, lower) = levels.split_at_mut(l + 1);
        let (q, qw, qh) = &mut upper[l];
        let (p, pw) = (&lower[0].0, lower[0].1);
        let (qw, qh) = (*qw, *qh);
        let todo: Vec<usize> = if l == 0 { cells.to_vec() } else { (0..qw * qh).filter(|&i| q[i].is_nan()).collect() };
        for &i in &todo {
            q[i] = p[(i / qw / 2) * pw + (i % qw) / 2];
        }
        let t = Instant::now();
        relax(q, qw, qh, &todo, if l == 0 { 20_000 } else { 50 });
        if log {
            eprintln!("  level {l} {qw}x{qh}: {} px relaxed in {:.1?}", todo.len(), t.elapsed());
        }
    }
    for &i in cells {
        v[i] = levels[0].0[i];
    }
}

/// Over-relaxed Gauss-Seidel on `cells` towards the mean of their finite
/// 4-neighbours, until no cell moves by more than 0.5 mm or `sweeps` run out.
fn relax(v: &mut [f64], w: usize, h: usize, cells: &[usize], sweeps: usize) {
    const OMEGA: f64 = 1.9;
    for _ in 0..sweeps {
        let mut moved: f64 = 0.0;
        for &i in cells {
            let (mut sum, mut k) = (0.0, 0);
            for j in neighbours(i, w, h) {
                if v[j].is_finite() {
                    sum += v[j];
                    k += 1;
                }
            }
            if k > 0 {
                let d = (sum / k as f64 - v[i]) * OMEGA;
                v[i] += d;
                moved = moved.max(d.abs());
            }
        }
        // A quarter of the 2 mm output step.
        if moved < 5e-4 {
            break;
        }
    }
}
