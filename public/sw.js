/**
 * ColorVision service worker: makes the app work fully offline.
 *
 * At build time `scripts/generate-sw.mjs` copies this file to `out/sw.js`
 * and fills in two placeholders:
 *   - CACHE_VERSION: a content hash of the build, so every deploy gets a
 *     fresh cache and the old one is deleted on activate.
 *   - PRECACHE_URLS: every file of the static export, including the 10 MB
 *     opencv.js. They're downloaded once at install, so the app works with
 *     no network from then on, even on the very first offline launch.
 *
 * Strategies:
 *   - Page navigations: network first (with a timeout), so updates are
 *     picked up when online. Falls back to the cached app shell when offline.
 *   - Everything else on our origin (hashed JS/CSS chunks, fonts, opencv.js,
 *     icons): cache first. These files never change for a given URL.
 *
 * In development the placeholders stay empty and the worker is never
 * registered (see components/ServiceWorkerRegistrar.tsx).
 */

const CACHE_VERSION = "__CACHE_VERSION__";
const PRECACHE_URLS = /* __PRECACHE_URLS__ */ [];

const CACHE_PREFIX = "colorvision-";
const CACHE_NAME = `${CACHE_PREFIX}${CACHE_VERSION}`;
/** After this long without a network answer, a navigation is served from cache. */
const NAVIGATION_TIMEOUT_MS = 4000;

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      // `cache: "reload"` bypasses the HTTP cache so we store fresh copies.
      await cache.addAll(PRECACHE_URLS.map((url) => new Request(url, { cache: "reload" })));
      // Activate right away. All assets are content-hashed, so a page still
      // running the old version keeps working.
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME)
          .map((key) => caches.delete(key)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return; // never touch third-party requests

  event.respondWith(request.mode === "navigate" ? networkFirst(request) : cacheFirst(request));
});

/** Network first with a timeout, falling back to the cached page or the app shell. */
async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await Promise.race([
      fetch(request),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("navigation timeout")), NAVIGATION_TIMEOUT_MS),
      ),
    ]);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch {
    const cached =
      (await cache.match(request, { ignoreSearch: true })) ?? (await cache.match("/"));
    return cached ?? Response.error();
  }
}

/** Cache first; anything missing is fetched and added to the cache. */
async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok && response.type === "basic") await cache.put(request, response.clone());
  return response;
}
