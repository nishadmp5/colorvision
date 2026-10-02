# ColorVision: Thread Matcher

An offline, single-screen PWA that helps a color-blind tailor find the thread that matches a piece of cloth, using the phone's rear camera.

1. Point the camera at the cloth and **tap it**. The cloth color is locked.
2. Show the threads. Each one is labeled live with **how many percent it matches** the cloth (e.g. "94% · THREAD 2"), and its box and label are colored by 10% band:

   | Band | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89 | 90–100 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | Color | dark purple | purple | indigo | blue | teal | sea green | green | light green | lime | bright yellow |

   The colors follow the *viridis* scale, which stays readable with color blindness: **brighter always means a closer match**, and box lines get thicker with each band. A color key sits above the buttons. The closest thread is marked **★**. Above the camera, the **top three threads** are shown as colored tiles, best first, with the closest one ringed in white.
3. With **VOICE GUIDANCE** on, the closest thread is spoken aloud ("Thread 2 is the closest match: 94 percent"). This happens again whenever the closest thread changes or moves into another 10% band. The phone also vibrates: two short pulses when the closest thread scores 75% or more, one long pulse when it's under 50%.

Status is never shown by color alone: every thread shows its percentage as text, and line thickness and brightness rise with the match.

The camera has its own area between the top and bottom bars, so nothing covers the picture. It asks for a **4:3** stream, the native shape of phone camera sensors, which gives the widest view. A 16:9 stream would crop the sensor. The video fills the area when that crops at most 25% of the frame; otherwise the whole frame is shown with thin black bars.

All processing happens on the device. There's no server, and after the first visit the app works without a network.

## How it works

| Piece | File |
|---|---|
| Color math: sRGB → XYZ (D65) → CIELAB, CIEDE2000 (ΔE₀₀), score = 100·e^(−0.1·ΔE₀₀) | [utils/colorMath.ts](utils/colorMath.ts) |
| Thread detection: OpenCV.js contour tree on L/a/b edges (background regions touching the frame edges are skipped), plus a pure-canvas fallback while OpenCV loads | [utils/spoolDetection.ts](utils/spoolDetection.ts) |
| OpenCV.js loader and typings | [utils/opencv.ts](utils/opencv.ts) |
| Speech and vibration | [utils/feedback.ts](utils/feedback.ts) |
| Camera (4:3, the sensor's full field of view), overlay, controls | [components/CameraScanner.tsx](components/CameraScanner.tsx), [components/Icons.tsx](components/Icons.tsx) |
| Offline service worker; its precache list is filled in after the build | [public/sw.js](public/sw.js), [scripts/generate-sw.mjs](scripts/generate-sw.mjs) |

The band colors and the 50% / 75% vibration thresholds are in `utils/colorMath.ts` (`MATCH_BAND_COLORS`, `GOOD_MATCH_MIN_SCORE`, `BEST_MATCH_MIN_SCORE`).

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
- **Background matters.** A thread that is nearly the same color as the surface it stands on has almost no visible outline, so it may not be detected. Stand threads on a plain, contrasting surface (a table or tray) and keep the cloth swatch in a separate part of the frame.
- Vibration isn't available on iOS. Speech works on iOS after VOICE GUIDANCE is switched on with a tap.
- The detection thresholds (`DEFAULT_DETECTION_OPTIONS`) were tuned on synthetic scenes and should be re-checked with real threads and phones.
