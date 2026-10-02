/**
 * Spool detection and cloth sampling.
 *
 * Everything works on a small, downscaled copy of the camera frame
 * (~360 px wide, see PROCESSING_WIDTH). That's plenty for color matching
 * and keeps each 300 ms tick fast on low-end phones.
 *
 * Two detectors produce the same `Region[]` output:
 *   - `detectRegionsOpenCv` does object contour segmentation with OpenCV.js
 *     (the primary engine).
 *   - `detectRegionsCanvas` is a pure-JS grid/flood-fill fallback, used while
 *     the 10 MB OpenCV engine is still loading or if it fails.
 *
 * `rankRegions` then scores each region against the locked cloth color with
 * CIEDE2000 and numbers the spools left to right.
 */
import {
  classifyMatch,
  deltaE2000,
  matchScore,
  rgbToLab,
  type Lab,
  type MatchLevel,
  type Rgb,
} from "./colorMath";
import type { CvMat, OpenCV } from "./opencv";

/** Width (px) of the downscaled analysis frame. */
export const PROCESSING_WIDTH = 360;

/** A point to exclude from detection (the cloth target), in processing-frame px. */
export interface ExcludePoint {
  x: number;
  y: number;
}

/** A detected candidate object, in processing-frame px. */
export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
  /** Robust (per-channel median) color of the object's interior. */
  rgb: Rgb;
}

/** A region scored against the cloth color. */
export interface Detection extends Region {
  /** 1-based spool number, assigned left to right so it stays stable between frames. */
  spool: number;
  lab: Lab;
  deltaE: number;
  /** 0–100 match percentage (100 · e^(−0.1·ΔE₀₀)). */
  score: number;
  level: MatchLevel;
}

/** Tunable detector thresholds. The defaults were tuned on 360 px frames. */
export interface DetectionOptions {
  /** Minimum object area as a fraction of the frame. */
  minAreaRatio: number;
  /** Maximum object area as a fraction of the frame (bigger = background / cloth). */
  maxAreaRatio: number;
  /** Maximum bounding-box aspect ratio (long side / short side). */
  maxAspect: number;
  /** Minimum solidity (contour area / convex hull area). */
  minSolidity: number;
  /** Maximum number of candidates returned (largest first). */
  maxRegions: number;
  /** Canny [low, high] hysteresis thresholds for the lightness (L) channel. */
  lightnessCanny: readonly [number, number];
  /** Canny [low, high] thresholds for the a/b channels (they have less range than L). */
  chromaCanny: readonly [number, number];
}

export const DEFAULT_DETECTION_OPTIONS: Readonly<DetectionOptions> = {
  minAreaRatio: 0.004,
  maxAreaRatio: 0.3,
  maxAspect: 5,
  minSolidity: 0.55,
  maxRegions: 8,
  lightnessCanny: [40, 100],
  chromaCanny: [20, 50],
};

/* -------------------------------------------------------------------------- */
/*                           Robust color sampling                            */
/* -------------------------------------------------------------------------- */

/**
 * Per-channel median RGB over the pixels selected by `include`. It uses
 * 256-bin histograms, so it's O(n) with no sorting. A median ignores specular
 * highlights, thread-groove shadows and stray edge pixels, which a mean
 * would pick up.
 *
 * @returns null when no pixel was selected.
 */
function medianRgb(
  img: ImageData,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  include: (x: number, y: number) => boolean,
): Rgb | null {
  const hr = new Uint32Array(256);
  const hg = new Uint32Array(256);
  const hb = new Uint32Array(256);
  const { data, width } = img;
  let n = 0;

  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (!include(x, y)) continue;
      const i = (y * width + x) * 4;
      hr[data[i]]++;
      hg[data[i + 1]]++;
      hb[data[i + 2]]++;
      n++;
    }
  }
  if (n === 0) return null;

  const median = (h: Uint32Array): number => {
    const half = n / 2;
    let acc = 0;
    for (let v = 0; v < 256; v++) {
      acc += h[v];
      if (acc >= half) return v;
    }
    return 255;
  };
  return { r: median(hr), g: median(hg), b: median(hb) };
}

/**
 * Sample the cloth color: the median over a disc around the tapped point.
 *
 * @param cx, cy  Disc center in processing-frame px.
 * @param radius  Disc radius in px (default: 3% of the frame width).
 */
export function sampleClothLab(
  img: ImageData,
  cx: number,
  cy: number,
  radius = Math.max(4, Math.round(img.width * 0.03)),
): Lab | null {
  const r2 = radius * radius;
  const x0 = Math.max(0, Math.floor(cx - radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const x1 = Math.min(img.width, Math.ceil(cx + radius + 1));
  const y1 = Math.min(img.height, Math.ceil(cy + radius + 1));

  const rgb = medianRgb(img, x0, y0, x1, y1, (x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r2);
  return rgb ? rgbToLab(rgb) : null;
}

/* -------------------------------------------------------------------------- */
/*                          OpenCV contour detector                           */
/* -------------------------------------------------------------------------- */

/**
 * Find object-like regions with OpenCV.js contour segmentation.
 *
 * Pipeline:
 *   RGBA → RGB → CIELAB → median blur (removes fabric texture)
 *   → Canny on each of L, a, b, OR-ed together. Edges in the color channels
 *     catch objects that differ from the cloth only in hue, not brightness,
 *     which is exactly the case a color-blind user can't judge.
 *   → dilate + morphological close (join broken outlines)
 *   → external contours, filtered on area / aspect ratio / solidity
 *   → per-contour filled mask, eroded so the edge halo is skipped
 *   → median color of the masked pixels.
 *
 * Every Mat is tracked and freed in `finally`. OpenCV.js memory lives on
 * the WebAssembly heap and is NOT garbage collected.
 */
export function detectRegionsOpenCv(
  cv: OpenCV,
  img: ImageData,
  exclude: ExcludePoint | null,
  options: Readonly<DetectionOptions> = DEFAULT_DETECTION_OPTIONS,
): Region[] {
  const owned: { delete(): void }[] = [];
  const own = <T extends { delete(): void }>(m: T): T => {
    owned.push(m);
    return m;
  };

  const { width: w, height: h } = img;
  const frameArea = w * h;

  try {
    const src = own(cv.matFromImageData(img));
    const rgb = own(new cv.Mat());
    cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
    const lab = own(new cv.Mat());
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    cv.medianBlur(lab, lab, 7);

    // Edge map: OR of the Canny edges of the L, a and b channels.
    const channels = own(new cv.MatVector());
    cv.split(lab, channels);
    const edges = own(cv.Mat.zeros(h, w, cv.CV_8UC1));
    const channelEdges = own(new cv.Mat());
    const cannyThresholds = [options.lightnessCanny, options.chromaCanny, options.chromaCanny];
    for (let c = 0; c < 3; c++) {
      const channel = channels.get(c);
      cv.Canny(channel, channelEdges, cannyThresholds[c][0], cannyThresholds[c][1]);
      cv.bitwise_or(edges, channelEdges, edges);
      channel.delete();
    }

    // Close small gaps so each object outline becomes one connected blob.
    const kernel = own(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5)));
    cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), 1);
    cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel);

    const contours = own(new cv.MatVector());
    const hierarchy = own(new cv.Mat());
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // --- Geometric filtering ---------------------------------------------
    const candidates: { index: number; area: number; rect: Omit<Region, "rgb"> }[] = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      let hull: CvMat | null = null;
      try {
        const area = cv.contourArea(contour);
        if (area < options.minAreaRatio * frameArea || area > options.maxAreaRatio * frameArea) {
          continue;
        }

        const rect = cv.boundingRect(contour);
        const aspect = Math.max(rect.width, rect.height) / Math.max(1, Math.min(rect.width, rect.height));
        if (aspect > options.maxAspect) continue;

        hull = new cv.Mat();
        cv.convexHull(contour, hull);
        const hullArea = cv.contourArea(hull);
        if (hullArea <= 0 || area / hullArea < options.minSolidity) continue;

        // The object the user tapped is the cloth, not a spool.
        if (
          exclude &&
          exclude.x >= rect.x &&
          exclude.x <= rect.x + rect.width &&
          exclude.y >= rect.y &&
          exclude.y <= rect.y + rect.height
        ) {
          continue;
        }

        candidates.push({
          index: i,
          area,
          rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        });
      } finally {
        hull?.delete();
        contour.delete();
      }
    }

    candidates.sort((a, b) => b.area - a.area);

    // --- Color sampling inside each object ----------------------------------
    const erodeKernel = own(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7)));
    const regions: Region[] = [];
    for (const cand of candidates.slice(0, options.maxRegions)) {
      const mask = cv.Mat.zeros(h, w, cv.CV_8UC1);
      try {
        cv.drawContours(mask, contours, cand.index, new cv.Scalar(255), -1);
        // Shrink the filled shape so the dilated edge halo / background isn't sampled.
        cv.erode(mask, mask, erodeKernel);
        const { x, y, width, height } = cand.rect;
        const maskData = mask.data;
        const color = medianRgb(img, x, y, x + width, y + height, (px, py) => maskData[py * w + px] !== 0);
        if (color) regions.push({ ...cand.rect, rgb: color });
      } finally {
        mask.delete();
      }
    }
    return regions;
  } finally {
    for (const m of owned) m.delete();
  }
}

/* -------------------------------------------------------------------------- */
/*                         Canvas fallback detector                           */
/* -------------------------------------------------------------------------- */

/**
 * Pure-JS fallback: split the frame into a grid of cells, then merge
 * neighbouring uniform cells of similar color (ΔE₀₀ < 5) into blobs with a
 * flood fill. Large blobs and blobs touching two or more frame edges are
 * treated as background (the cloth or the table). It's less precise than
 * OpenCV, but enough to give feedback within the first second.
 */
export function detectRegionsCanvas(
  img: ImageData,
  exclude: ExcludePoint | null,
  options: Readonly<DetectionOptions> = DEFAULT_DETECTION_OPTIONS,
): Region[] {
  // ~15 px cells at 360 px: a spool spans at least 3 cells, so its interior
  // has 2+ uniform cells even when its edges straddle cell borders.
  const COLS = 24;
  const rows = Math.max(4, Math.round((COLS * img.height) / img.width));
  const cw = img.width / COLS;
  const ch = img.height / rows;
  const { data, width } = img;

  interface Cell {
    rgb: Rgb;
    lab: Lab;
    uniform: boolean;
  }
  const cells: Cell[] = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < COLS; c++) {
      const x0 = Math.floor(c * cw);
      const y0 = Math.floor(r * ch);
      const x1 = Math.floor((c + 1) * cw);
      const y1 = Math.floor((r + 1) * ch);
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let sl = 0;
      let sl2 = 0;
      let n = 0;
      // Every second pixel is enough for a cell average.
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const i = (y * width + x) * 4;
          const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          sr += data[i];
          sg += data[i + 1];
          sb += data[i + 2];
          sl += lum;
          sl2 += lum * lum;
          n++;
        }
      }
      const rgb = { r: sr / n, g: sg / n, b: sb / n };
      const std = Math.sqrt(Math.max(0, sl2 / n - (sl / n) ** 2));
      cells.push({ rgb, lab: rgbToLab(rgb), uniform: std < 28 });
    }
  }

  const excludeCell = exclude
    ? Math.min(rows - 1, Math.floor(exclude.y / ch)) * COLS + Math.min(COLS - 1, Math.floor(exclude.x / cw))
    : -1;

  const visited = new Uint8Array(cells.length);
  const regions: (Region & { size: number })[] = [];

  for (let start = 0; start < cells.length; start++) {
    if (visited[start] || !cells[start].uniform) continue;

    // Flood fill over 4-connected neighbours with similar color.
    const stack = [start];
    const members: number[] = [];
    visited[start] = 1;
    while (stack.length) {
      const idx = stack.pop()!;
      members.push(idx);
      const r = Math.floor(idx / COLS);
      const c = idx % COLS;
      const neighbours = [
        r > 0 ? idx - COLS : -1,
        r < rows - 1 ? idx + COLS : -1,
        c > 0 ? idx - 1 : -1,
        c < COLS - 1 ? idx + 1 : -1,
      ];
      for (const nb of neighbours) {
        if (nb < 0 || visited[nb] || !cells[nb].uniform) continue;
        if (deltaE2000(cells[idx].lab, cells[nb].lab) < 5) {
          visited[nb] = 1;
          stack.push(nb);
        }
      }
    }

    if (members.length < 2 || members.length > options.maxAreaRatio * cells.length) continue;
    if (members.includes(excludeCell)) continue;

    let minR = rows;
    let maxR = 0;
    let minC = COLS;
    let maxC = 0;
    let sr = 0;
    let sg = 0;
    let sb = 0;
    for (const m of members) {
      const r = Math.floor(m / COLS);
      const c = m % COLS;
      minR = Math.min(minR, r);
      maxR = Math.max(maxR, r);
      minC = Math.min(minC, c);
      maxC = Math.max(maxC, c);
      sr += cells[m].rgb.r;
      sg += cells[m].rgb.g;
      sb += cells[m].rgb.b;
    }
    const edgesTouched =
      Number(minR === 0) + Number(maxR === rows - 1) + Number(minC === 0) + Number(maxC === COLS - 1);
    if (edgesTouched >= 2) continue; // Background, not an object.
    // A 1-cell-wide strip is a mixed edge (object shading + background), not an object.
    if (maxR - minR < 1 || maxC - minC < 1) continue;

    const n = members.length;
    regions.push({
      x: minC * cw,
      y: minR * ch,
      width: (maxC - minC + 1) * cw,
      height: (maxR - minR + 1) * ch,
      rgb: { r: Math.round(sr / n), g: Math.round(sg / n), b: Math.round(sb / n) },
      size: n,
    });
  }

  // Shading and thread grooves can split one spool into several blobs.
  // Merge blobs whose boxes overlap a lot (they can't be separate objects),
  // or that touch (within one cell) and have similar color.
  const gap = Math.max(cw, ch);
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < regions.length; i++) {
      for (let j = i + 1; j < regions.length; j++) {
        const a = regions[i];
        const b = regions[j];
        const touching =
          a.x <= b.x + b.width + gap &&
          b.x <= a.x + a.width + gap &&
          a.y <= b.y + b.height + gap &&
          b.y <= a.y + a.height + gap;
        if (!touching) continue;
        const overlapW = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
        const overlapH = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
        const overlapRatio =
          overlapW > 0 && overlapH > 0
            ? (overlapW * overlapH) / Math.min(a.width * a.height, b.width * b.height)
            : 0;
        if (overlapRatio < 0.3 && deltaE2000(rgbToLab(a.rgb), rgbToLab(b.rgb)) >= 12) continue;

        const n = a.size + b.size;
        const x = Math.min(a.x, b.x);
        const y = Math.min(a.y, b.y);
        regions[i] = {
          x,
          y,
          width: Math.max(a.x + a.width, b.x + b.width) - x,
          height: Math.max(a.y + a.height, b.y + b.height) - y,
          // Size-weighted average color.
          rgb: {
            r: Math.round((a.rgb.r * a.size + b.rgb.r * b.size) / n),
            g: Math.round((a.rgb.g * a.size + b.rgb.g * b.size) / n),
            b: Math.round((a.rgb.b * a.size + b.rgb.b * b.size) / n),
          },
          size: n,
        };
        regions.splice(j, 1);
        merged = true;
        break outer;
      }
    }
  }

  return regions
    .sort((a, b) => b.size - a.size)
    .slice(0, options.maxRegions)
    .map(({ x, y, width, height, rgb }) => ({ x, y, width, height, rgb }));
}

/* -------------------------------------------------------------------------- */
/*                                  Ranking                                   */
/* -------------------------------------------------------------------------- */

/**
 * Score regions against the cloth color and assign spool numbers.
 *
 * - Spools are numbered **left to right** (by box center), so "Spool 1"
 *   means the same physical spool from one frame to the next.
 * - Only the single highest-scoring spool can be BEST. Other spools above
 *   the BEST threshold are shown as GOOD, so there's never more than one
 *   star on screen.
 */
export function rankRegions(regions: Region[], clothLab: Lab): Detection[] {
  const detections: Detection[] = regions
    .slice()
    .sort((a, b) => a.x + a.width / 2 - (b.x + b.width / 2))
    .map((region, i) => {
      const lab = rgbToLab(region.rgb);
      const deltaE = deltaE2000(clothLab, lab);
      const score = matchScore(deltaE);
      return { ...region, spool: i + 1, lab, deltaE, score, level: classifyMatch(score) };
    });

  const top = detections.reduce<Detection | null>(
    (best, d) => (!best || d.score > best.score ? d : best),
    null,
  );
  for (const d of detections) {
    if (d.level === "best" && d !== top) d.level = "good";
  }
  return detections;
}
