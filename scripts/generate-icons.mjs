#!/usr/bin/env node
/**
 * Generate the PWA icons (PNG) with no dependencies: shapes are rasterised
 * with 4×4 supersampling and encoded with a minimal PNG writer
 * (zlib + CRC-32 from Node core).
 *
 * Artwork: a thread spool (white flanges, gold thread) with a bright green
 * check badge, on black. Run once and commit the output:
 *
 *   node scripts/generate-icons.mjs
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { crc32, deflateSync } from "node:zlib";

const outDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "icons");

const BLACK = [0, 0, 0];
const WHITE = [255, 255, 255];
const GOLD = [255, 215, 0];
const THREAD_LINE = [176, 140, 0];
const GREEN = [0, 255, 0];

/* ------------------------------ Geometry ------------------------------ */

const roundRect = (x0, y0, x1, y1, r) => (u, v) => {
  if (u < x0 || u > x1 || v < y0 || v > y1) return false;
  const cx = Math.min(Math.max(u, x0 + r), x1 - r);
  const cy = Math.min(Math.max(v, y0 + r), y1 - r);
  return (u - cx) ** 2 + (v - cy) ** 2 <= r * r;
};
const circle = (cx, cy, r) => (u, v) => (u - cx) ** 2 + (v - cy) ** 2 <= r * r;
/** Thick line segment (capsule). */
const segment = (ax, ay, bx, by, halfWidth) => (u, v) => {
  const dx = bx - ax;
  const dy = by - ay;
  const t = Math.max(0, Math.min(1, ((u - ax) * dx + (v - ay) * dy) / (dx * dx + dy * dy)));
  return (u - ax - t * dx) ** 2 + (v - ay - t * dy) ** 2 <= halfWidth * halfWidth;
};

/** Layers in paint order: [test(u, v), color]. Coordinates are 0–1. */
function artwork() {
  const body = roundRect(0.3, 0.24, 0.7, 0.76, 0.02);
  return [
    [() => true, BLACK],
    [body, GOLD],
    // Thread windings: thin darker bands across the body.
    [(u, v) => body(u, v) && (v - 0.24) % 0.06 < 0.014, THREAD_LINE],
    [roundRect(0.2, 0.15, 0.8, 0.25, 0.03), WHITE], // top flange
    [roundRect(0.2, 0.75, 0.8, 0.85, 0.03), WHITE], // bottom flange
    // Match badge: green disc with a black check, outlined in black.
    [circle(0.74, 0.72, 0.19), BLACK],
    [circle(0.74, 0.72, 0.16), GREEN],
    [segment(0.66, 0.72, 0.715, 0.78, 0.026), BLACK],
    [segment(0.715, 0.78, 0.83, 0.65, 0.026), BLACK],
  ];
}

/**
 * Rasterise to RGBA. `scale` < 1 shrinks the artwork around the center
 * (maskable icons must keep content inside the 80% safe zone).
 */
function render(size, scale = 1) {
  const layers = artwork();
  const SS = 4;
  const rgba = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const u = ((x + (sx + 0.5) / SS) / size - 0.5) / scale + 0.5;
          const v = ((y + (sy + 0.5) / SS) / size - 0.5) / scale + 0.5;
          let color = BLACK;
          for (const [test, c] of layers) if (test(u, v)) color = c;
          r += color[0];
          g += color[1];
          b += color[2];
        }
      }
      const i = (y * size + x) * 4;
      const n = SS * SS;
      rgba[i] = Math.round(r / n);
      rgba[i + 1] = Math.round(g / n);
      rgba[i + 2] = Math.round(b / n);
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

/* ------------------------------ PNG writer ----------------------------- */

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const typeAndData = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(typeAndData) >>> 0);
  return Buffer.concat([len, typeAndData, crc]);
}

function encodePng(size, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type RGBA
  // compression, filter and interlace stay 0.

  // Each scanline is prefixed with filter type 0 (None).
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    rgba.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/* -------------------------------- Output ------------------------------- */

const targets = [
  { file: "icon-192.png", size: 192, scale: 1 },
  { file: "icon-512.png", size: 512, scale: 1 },
  { file: "maskable-512.png", size: 512, scale: 0.72 },
  { file: "apple-touch-icon.png", size: 180, scale: 0.9 },
  { file: "favicon-32.png", size: 32, scale: 1 },
];

mkdirSync(outDir, { recursive: true });
for (const { file, size, scale } of targets) {
  writeFileSync(join(outDir, file), encodePng(size, render(size, scale)));
  console.log(`[icons] public/icons/${file} (${size}×${size})`);
}
