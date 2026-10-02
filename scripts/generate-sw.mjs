#!/usr/bin/env node
/**
 * Post-build step: fill in the service worker's precache manifest.
 *
 * Runs automatically after `next build` (the npm `postbuild` script). It:
 *   1. lists every file in the static export (`out/`),
 *   2. maps it to the URL it's served at (`index.html` → `/`),
 *   3. hashes the whole build into a cache version,
 *   4. writes both into `out/sw.js` (copied there from `public/sw.js`).
 *
 * This replaces `next-pwa`, which only works with webpack. Next 16 builds
 * with Turbopack.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const outDir = join(root, "out");
const swPath = join(outDir, "sw.js");

if (!existsSync(swPath)) {
  console.error("[sw] out/sw.js not found. Run `next build` (with output: 'export') first.");
  process.exit(1);
}

/** Files that must never be precached. */
const EXCLUDE = [/^sw\.js$/, /\.map$/, /(^|\/)\.DS_Store$/];

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    return statSync(full).isDirectory() ? walk(full) : [full];
  });
}

const files = walk(outDir)
  .map((full) => ({ full, rel: relative(outDir, full).split(sep).join("/") }))
  .filter(({ rel }) => !EXCLUDE.some((re) => re.test(rel)))
  .sort((a, b) => a.rel.localeCompare(b.rel));

const hash = createHash("sha256");
const urls = files.map(({ full, rel }) => {
  hash.update(rel);
  hash.update(readFileSync(full));
  // The home page is requested as "/" (start_url); other files by their own path.
  return rel === "index.html" ? "/" : `/${rel}`;
});
const version = hash.digest("hex").slice(0, 12);

let sw = readFileSync(swPath, "utf8");
const VERSION_TOKEN = '"__CACHE_VERSION__"';
const URLS_TOKEN = "/* __PRECACHE_URLS__ */ []";
if (!sw.includes(VERSION_TOKEN) || !sw.includes(URLS_TOKEN)) {
  console.error("[sw] Placeholders not found in out/sw.js. Has public/sw.js been changed?");
  process.exit(1);
}
sw = sw.replace(VERSION_TOKEN, JSON.stringify(version)).replace(URLS_TOKEN, JSON.stringify(urls, null, 2));
writeFileSync(swPath, sw);

const totalBytes = files.reduce((n, { full }) => n + statSync(full).size, 0);
console.log(
  `[sw] Precache manifest: ${urls.length} files, ${(totalBytes / 1e6).toFixed(1)} MB, version ${version}`,
);
