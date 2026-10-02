#!/usr/bin/env node
/**
 * Make sure `public/opencv.js` exists before `next dev` / `next build`.
 *
 * OpenCV.js is a ~10 MB single-file build (JS plus inlined WebAssembly), so it
 * is not committed to git. This script downloads a pinned, official release
 * once. After that it's served from our own origin and precached by the
 * service worker, so the *app* never needs the network.
 *
 * Override the source with OPENCV_JS_URL=<url> (e.g. an internal mirror).
 */
import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** Pinned official build. Version-pinned on purpose: the API surface we use is stable in 4.x. */
const OPENCV_VERSION = "4.9.0";
const SOURCE_URL =
  process.env.OPENCV_JS_URL ?? `https://docs.opencv.org/${OPENCV_VERSION}/opencv.js`;

/** Anything smaller than this is certainly not a real OpenCV build (e.g. an HTML error page). */
const MIN_BYTES = 5 * 1024 * 1024;

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "public", "opencv.js");

if (existsSync(target) && statSync(target).size >= MIN_BYTES) {
  console.log(`[opencv] public/opencv.js present (${(statSync(target).size / 1e6).toFixed(1)} MB)`);
  process.exit(0);
}

console.log(`[opencv] Downloading OpenCV.js ${OPENCV_VERSION} from ${SOURCE_URL} …`);
try {
  const res = await fetch(SOURCE_URL);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

  const body = Buffer.from(await res.arrayBuffer());
  if (body.length < MIN_BYTES) {
    throw new Error(`downloaded file is only ${body.length} bytes — not a valid OpenCV.js build`);
  }

  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, body);
  console.log(`[opencv] Saved public/opencv.js (${(body.length / 1e6).toFixed(1)} MB)`);
} catch (err) {
  console.error(`[opencv] Download failed: ${err instanceof Error ? err.message : err}`);
  console.error("[opencv] Place an opencv.js build at public/opencv.js manually, or set OPENCV_JS_URL.");
  process.exit(1);
}
