/**
 * Next.js configuration for ColorVision.
 *
 * The app is a 100% client-side PWA: no server, no API routes, no runtime
 * image optimisation. `output: 'export'` produces a plain static site in
 * `out/` that can be served from any static host (or straight from the
 * service worker cache when offline).
 *
 * Note: the classic `next-pwa` plugin only hooks into webpack, while Next 16
 * builds with Turbopack. Offline support is therefore provided by our own
 * dependency-free service worker (`public/sw.js`), whose precache manifest is
 * injected after the build by `scripts/generate-sw.mjs`.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  // Emit a fully static site to `out/`.
  output: "export",

  // The Image Optimization API needs a server; serve images as-is instead.
  images: {
    unoptimized: true,
  },

  // The floating dev-tools badge sits over the bottom-left of the screen,
  // right on top of the RESET button, which gets in the way when testing on a phone.
  devIndicators: false,
};

export default nextConfig;
