# ColorVision: Thread Matcher

An offline, single-screen PWA that helps a color-blind tailor find the thread spool that matches a piece of cloth, using the phone's rear camera.

1. Point the camera at the cloth and **tap it**. The cloth color is locked.
2. Show the spools. Each one is labeled live:
   - **★ BEST MATCH**: solid green box, gold label
   - **✓ GOOD MATCH**: dashed yellow box
   - **✗ NO MATCH**: thin gray box
3. With **VOICE GUIDANCE** on, results are spoken aloud ("Best match found! Spool 2 is a 91% match"). The phone also vibrates: two short pulses for a best match, one long pulse for no match.

Status is never shown by color alone. Every state has its own icon, line style and words.

All processing happens on the device. There's no server, and after the first visit the app works without a network.

## How it works

| Piece | File |
|---|---|
| Color math: sRGB → XYZ (D65) → CIELAB, CIEDE2000 (ΔE₀₀), score = 100·e^(−0.1·ΔE₀₀) | [utils/colorMath.ts](utils/colorMath.ts) |
| Spool detection: OpenCV.js contours on L/a/b edges, plus a pure-canvas fallback while OpenCV loads | [utils/spoolDetection.ts](utils/spoolDetection.ts) |
| OpenCV.js loader and typings | [utils/opencv.ts](utils/opencv.ts) |
| Speech and vibration | [utils/feedback.ts](utils/feedback.ts) |
| Camera, overlay, controls | [components/CameraScanner.tsx](components/CameraScanner.tsx) |
| Offline service worker; its precache list is filled in after the build | [public/sw.js](public/sw.js), [scripts/generate-sw.mjs](scripts/generate-sw.mjs) |

A spool is a **BEST** match when its score is ≥ 75% (ΔE₀₀ ≲ 2.9) and it's the top-scoring spool. It's a **GOOD** match at ≥ 55% (ΔE₀₀ ≲ 6). These thresholds are in `utils/colorMath.ts`.

## Development

```bash
npm install
npm run dev        # downloads public/opencv.js on first run (~10 MB)
```

The camera needs a secure context. `http://localhost` works. On a phone, use HTTPS: a tunnel such as `cloudflared` or `ngrok`, or a LAN certificate.

| Script | What it does |
|---|---|
| `npm run build` | Fetches OpenCV.js if missing, runs `next build` (static export to `out/`), then writes the precache manifest into `out/sw.js` |
| `npm start` | Serves `out/` locally |
| `npm run lint`, `npm run typecheck` | ESLint, TypeScript |
| `node scripts/generate-icons.mjs` | Regenerates the PNG icons in `public/icons/` |

`public/opencv.js` (OpenCV 4.9.0) is downloaded by `scripts/fetch-opencv.mjs` and isn't committed. To use a mirror, set `OPENCV_JS_URL`.

## Deploying

Upload `out/` to any static HTTPS host. Serve `sw.js` with `Cache-Control: no-cache` so updates are picked up. The first visit precaches the whole app, about 11 MB.

Next 16 builds with Turbopack and the classic `next-pwa` plugin only works with webpack, so offline support comes from the hand-written service worker instead.

## Known limitations

- **Lighting changes colors.** Use even light; the 🔦 FLASH LIGHT button helps where the phone supports a torch (most Android phones; not iOS Safari).
- **Background matters.** A spool that is nearly the same color as the surface it stands on has almost no visible outline, so it may not be detected. Stand spools on a plain, contrasting surface (a table or tray) and keep the cloth swatch in a separate part of the frame.
- Vibration isn't available on iOS. Speech works on iOS after VOICE GUIDANCE is switched on with a tap.
- The detection thresholds (`DEFAULT_DETECTION_OPTIONS`) were tuned on synthetic scenes and should be re-checked with real spools and phones.
