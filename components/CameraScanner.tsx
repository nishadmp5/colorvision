"use client";

/**
 * CameraScanner: the single screen of ColorVision.
 *
 * ┌───────────────────────────────────────────┐
 * │ [🔒 CLOTH LOCKED ■]          [FLASH OFF] │  top bar
 * │ [ 94% ] ★ CLOSEST MATCH · Spool 2        │  live result card (aria-live)
 * │ ╭───────────────────────────────────────╮ │
 * │ │  live rear camera (4:3) + overlay     │ │  camera area: no bars on top,
 * │ │  tap = pick the cloth                 │ │  boxes colored by 10% band
 * │ ╰───────────────────────────────────────╯ │
 * │ 0% ▮▮▮▮▮▮▮▮▮▮ 100%                         │  color key
 * │ ( ↺ RESET CLOTH COLOR )  ( 🔊 VOICE ON )  │  bottom bar
 * └───────────────────────────────────────────┘
 *
 * Data flow, every PROCESS_INTERVAL_MS (300 ms):
 *   video frame → downscaled ImageData (~360 px wide)
 *     → (if a tap is pending) sample + lock the cloth Lab color
 *     → detect spools (OpenCV.js contours, or the canvas fallback while loading)
 *     → CIEDE2000 score for each spool → draw boxes → update banner
 *     → speak / vibrate when a stable result changes
 *
 * Accessibility rules followed throughout:
 *   - Match level is never shown by color alone: every spool shows its
 *     percentage as text, and box lines get thicker and brighter (viridis
 *     scale) with each 10% band. Controls always have text next to their icon.
 *   - Touch targets are ≥ 64 px tall, with 20–28 px bold text.
 *   - UI chrome uses black, white, #FFD700 and #FF0000; match bands use viridis.
 *
 * This component touches `window`, `navigator.mediaDevices` and the canvas,
 * so it is client-only. `app/page.tsx` loads it with `ssr: false`.
 */
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import {
  MATCH_BAND_COLORS,
  labToCssRgb,
  matchBandLabel,
  readableTextColor,
  type Lab,
  type MatchLevel,
} from "@/utils/colorMath";
import {
  HAPTIC_BEST_MATCH,
  HAPTIC_NO_MATCH,
  speak,
  stopSpeaking,
  vibrate,
} from "@/utils/feedback";
import { waitForOpenCv, type OpenCV } from "@/utils/opencv";
import {
  AlertIcon,
  CameraIcon,
  FlashlightIcon,
  FlashlightOffIcon,
  LockIcon,
  ResetIcon,
  ScanIcon,
  StarIcon,
  TargetIcon,
  UnlockIcon,
  VolumeOffIcon,
  VolumeOnIcon,
} from "@/components/Icons";
import {
  PROCESSING_WIDTH,
  detectRegionsCanvas,
  detectRegionsOpenCv,
  rankRegions,
  sampleClothLab,
  type Detection,
  type Region,
} from "@/utils/spoolDetection";

/* -------------------------------------------------------------------------- */
/*                                  Constants                                 */
/* -------------------------------------------------------------------------- */

/** How often a frame is analysed. */
const PROCESS_INTERVAL_MS = 300;
/** A result must hold for this many consecutive frames before it's announced. */
const ANNOUNCE_STABLE_FRAMES = 2;
/** Minimum time between two spoken/vibrated announcements. */
const ANNOUNCE_COOLDOWN_MS = 4000;
/** localStorage key for the voice guidance preference. */
const VOICE_STORAGE_KEY = "colorvision.voice";

/** Overlay palette (matches the CSS tokens in globals.css). */
const COLORS = {
  black: "#000000",
  white: "#FFFFFF",
  yellow: "#FFD700",
  green: "#00FF00",
  gray: "#8A8A8A",
} as const;

/* -------------------------------------------------------------------------- */
/*                                    Types                                   */
/* -------------------------------------------------------------------------- */

type CameraState =
  | { kind: "starting" }
  | { kind: "running" }
  | { kind: "error"; message: string };

/** Which detector is active: OpenCV, the canvas fallback while loading, or the fallback for good. */
type Engine = "loading" | "opencv" | "basic";

/** Torch isn't in TypeScript's DOM typings yet (it's in the Image Capture spec). */
interface TorchCapabilities extends MediaTrackCapabilities {
  torch?: boolean;
}
interface TorchConstraintSet extends MediaTrackConstraintSet {
  torch?: boolean;
}

/** A locked cloth color plus where it was sampled (normalised 0–1 video coordinates). */
interface ClothTarget {
  lab: Lab;
  u: number;
  v: number;
}

/** A tap ripple animation instance, in CSS px relative to the screen. */
interface Ripple {
  id: number;
  x: number;
  y: number;
}

type SummaryLevel = MatchLevel | "idle" | "searching";

/** Banner text, speech text and a de-duplication key for announcements. */
interface Summary {
  level: SummaryLevel;
  /** Match band of the closest spool (sets the banner color); null before results. */
  band: number | null;
  /** The closest spool and its rounded percentage; null before results. */
  spool: number | null;
  pct: number | null;
  text: string;
  speech: string | null;
  key: string;
}

/** Mutable announcement bookkeeping (kept in a ref, not state). */
interface AnnouncerState {
  candidate: string;
  streak: number;
  announced: string;
  lastAt: number;
}

/** Maps video pixels to screen pixels, plus the CSS `object-fit` that matches it. */
interface FitTransform {
  scale: number;
  dx: number;
  dy: number;
  mode: "cover" | "contain";
}

/* -------------------------------------------------------------------------- */
/*                               Pure helpers                                 */
/* -------------------------------------------------------------------------- */

/** Largest share of the camera frame "cover" may crop away before we show the whole frame instead. */
const MAX_COVER_CROP = 0.25;

/**
 * Fit the camera frame into the camera area.
 *   - "cover" fills the area edge to edge and crops the overflow.
 *   - "contain" shows the entire frame, with thin black bars.
 * Cover is used when it crops at most 25% of the frame (a 4:3 portrait
 * stream loses under 10% on a tall phone, ~23% on a small one). Otherwise contain is used,
 * so a big part of what the camera sees is never hidden (e.g. a landscape
 * webcam in a portrait window). The same math draws the boxes exactly over
 * the objects and maps taps back into the video frame.
 */
function fitTransform(cw: number, ch: number, vw: number, vh: number): FitTransform {
  const cover = Math.max(cw / vw, ch / vh);
  const contain = Math.min(cw / vw, ch / vh);
  const visibleShare = (cw * ch) / (vw * vh * cover * cover);
  const mode = 1 - visibleShare <= MAX_COVER_CROP ? "cover" : "contain";
  const scale = mode === "cover" ? cover : contain;
  return { scale, dx: (cw - vw * scale) / 2, dy: (ch - vh * scale) / 2, mode };
}

/**
 * Open the rear camera at 1080p in **4:3**, stepping down the constraints
 * when the device can't satisfy them (e.g. laptops only have a front
 * camera): exact rear camera → preferred rear camera → any camera.
 *
 * 4:3 is the native shape of phone camera sensors, so it gives the widest
 * field of view. A 16:9 stream crops the top and bottom of the sensor
 * (about 25% of the picture).
 */
async function openCamera(): Promise<MediaStream> {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new DOMException("Camera needs HTTPS", "SecurityError");
  }
  const resolution: MediaTrackConstraints = {
    width: { ideal: 1440 },
    height: { ideal: 1080 },
    aspectRatio: { ideal: 4 / 3 },
  };
  const attempts: MediaTrackConstraints[] = [
    { ...resolution, facingMode: { exact: "environment" } },
    { ...resolution, facingMode: { ideal: "environment" } },
    resolution,
  ];

  let lastError: unknown = null;
  for (const video of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      lastError = err;
      // A denied permission won't change with other constraints: stop here.
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") break;
    }
  }
  throw lastError;
}

/** Turn getUserMedia errors into plain-language instructions. */
function describeCameraError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
      return "The camera is blocked. Please allow the camera for this app in your phone settings, then tap TRY AGAIN.";
    case "SecurityError":
      return "The camera only works on a secure (https) page. Please open the app from its https address.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found on this device.";
    case "NotReadableError":
      return "The camera is being used by another app. Close that app, then tap TRY AGAIN.";
    default:
      return "The camera could not start. Tap TRY AGAIN.";
  }
}

function stopStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop());
}

/** Build the banner text, spoken text and announcement key for a frame. */
function summarize(detections: Detection[], hasCloth: boolean): Summary {
  if (!hasCloth) {
    return { level: "idle", band: null, spool: null, pct: null, text: "Tap the cloth to lock its color", speech: null, key: "idle" };
  }
  if (detections.length === 0) {
    return { level: "searching", band: null, spool: null, pct: null, text: "Now show the threads", speech: null, key: "empty" };
  }

  const top = detections.find((d) => d.isTop) ?? detections[0];
  const pct = Math.round(top.score);
  return {
    // The level only picks the vibration pattern (best / no match).
    level: top.level,
    band: top.band,
    spool: top.spool,
    pct,
    text: `Closest match: thread ${top.spool}, ${pct}%`,
    speech: `Thread ${top.spool} is the closest match: ${pct} percent.`,
    // Re-announce when the closest spool changes or moves to another 10% band.
    key: `top:${top.spool}:${top.band}`,
  };
}

const freshAnnouncer = (): AnnouncerState => ({
  candidate: "",
  streak: 0,
  announced: "",
  lastAt: 0,
});

/** Read the saved voice preference; storage can throw (private mode, blocked site data). */
function loadVoicePreference(): boolean {
  try {
    return window.localStorage.getItem(VOICE_STORAGE_KEY) === "on";
  } catch {
    return false;
  }
}

function saveVoicePreference(on: boolean): void {
  try {
    window.localStorage.setItem(VOICE_STORAGE_KEY, on ? "on" : "off");
  } catch {
    // Not persisted: fine, the toggle still works for this session.
  }
}

/* -------------------------------------------------------------------------- */
/*                               Canvas drawing                               */
/* -------------------------------------------------------------------------- */

interface ScreenRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The visible strip between the top and bottom bars, in CSS px. */
interface DrawBounds {
  width: number;
  top: number;
  bottom: number;
}

interface BadgeStyle {
  bg: string;
  fg: string;
  border: string;
  fontPx: number;
}

/** One way of writing a label: its lines of text and its style. */
interface BadgeSpec {
  lines: string[];
  style: BadgeStyle;
}

/** A label to place: what it belongs to, the variants to try (best first), and whether it must be shown. */
interface LabelRequest {
  anchor: ScreenRect;
  variants: BadgeSpec[];
  force: boolean;
}

const CLOTH_BADGE_STYLE: BadgeStyle = { bg: COLORS.white, fg: COLORS.black, border: COLORS.black, fontPx: 20 };

/** A spool label filled with its band color; text is black or white, whichever reads better. */
function bandBadgeStyle(d: Detection): BadgeStyle {
  const bg = MATCH_BAND_COLORS[d.band];
  const fg = readableTextColor(bg);
  return { bg, fg, border: fg === COLORS.black ? COLORS.black : COLORS.white, fontPx: d.isTop ? 28 : 26 };
}

const BADGE_PAD_X = 10;
const BADGE_PAD_Y = 6;
const BADGE_GAP = 6;

/**
 * Label wording for each spool, longest first. The match percentage is the
 * main information; the fill color shows its 10% band. The closest spool also
 * gets a ★. When spools stand close together the label shrinks to the
 * percentage alone.
 */
function badgeVariants(d: Detection): BadgeSpec[] {
  const pct = `${d.isTop ? "★ " : ""}${Math.round(d.score)}%`;
  const style = bandBadgeStyle(d);
  return [
    { lines: [pct, `THREAD ${d.spool}`], style },
    { lines: [pct], style },
  ];
}

function measureBadge(ctx: CanvasRenderingContext2D, spec: BadgeSpec, font: string) {
  ctx.font = `700 ${spec.style.fontPx}px ${font}`;
  const lineH = Math.round(spec.style.fontPx * 1.2);
  const textW = Math.max(...spec.lines.map((l) => ctx.measureText(l).width));
  return { w: Math.ceil(textW) + BADGE_PAD_X * 2, h: lineH * spec.lines.length + BADGE_PAD_Y * 2 };
}

function paintBadge(ctx: CanvasRenderingContext2D, r: ScreenRect, spec: BadgeSpec, font: string): void {
  const { style } = spec;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.roundRect(r.x, r.y, r.w, r.h, 14);
  // Soft drop shadow lifts the label off the camera picture.
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  ctx.shadowBlur = 14;
  ctx.shadowOffsetY = 4;
  ctx.fillStyle = style.bg;
  ctx.fill();
  ctx.restore();
  ctx.strokeStyle = style.border;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.font = `700 ${style.fontPx}px ${font}`;
  ctx.fillStyle = style.fg;
  ctx.textBaseline = "top";
  const lineH = Math.round(style.fontPx * 1.2);
  spec.lines.forEach((line, i) => ctx.fillText(line, r.x + BADGE_PAD_X, r.y + BADGE_PAD_Y + i * lineH + 1));
}

const overlaps = (a: ScreenRect, b: ScreenRect, margin = 4): boolean =>
  a.x < b.x + b.w + margin && b.x < a.x + a.w + margin && a.y < b.y + b.h + margin && b.y < a.y + a.h + margin;

/** Size of the area two rectangles share (0 when they don't touch). */
const overlapArea = (a: ScreenRect, b: ScreenRect): number =>
  Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
  Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));

/** Candidate top-left corners for a w×h label next to its box, in order of preference. */
function labelSpots(a: ScreenRect, w: number, h: number): [number, number][] {
  const centerX = a.x + a.w / 2 - w / 2;
  return [
    [a.x, a.y - h - BADGE_GAP], // above, left-aligned
    [centerX, a.y - h - BADGE_GAP], // above, centered
    [a.x, a.y + a.h + BADGE_GAP], // below, left-aligned
    [centerX, a.y + a.h + BADGE_GAP], // below, centered
    [a.x + 4, a.y + 4], // inside, top
    [centerX, a.y + a.h / 2 - h / 2], // inside, middle
    [centerX, a.y + a.h - h - 4], // inside, bottom
  ];
}

/**
 * Greedy label layout. Requests are handled in priority order. Each one
 * tries its wording variants × positions (above, below, then inside its
 * box), clamped to the visible area, and takes the first spot that covers
 * neither a placed label nor another spool's box (then, failing that, the
 * first spot that only avoids labels). A `force`d label (every spool's
 * percentage) is always shown: if no spot is free, it takes its most compact
 * wording at the spot that covers the least of the other labels.
 */
function layoutLabels(
  ctx: CanvasRenderingContext2D,
  requests: LabelRequest[],
  obstacles: ScreenRect[],
  bounds: DrawBounds,
  font: string,
): { rect: ScreenRect; spec: BadgeSpec }[] {
  const placed: { rect: ScreenRect; spec: BadgeSpec }[] = [];

  const clamp = (x: number, y: number, w: number, h: number): ScreenRect => ({
    x: Math.min(Math.max(4, x), bounds.width - w - 4),
    y: Math.min(Math.max(bounds.top + 4, y), bounds.bottom - h - 4),
    w,
    h,
  });

  for (const req of requests) {
    const a = req.anchor;
    let chosen: { rect: ScreenRect; spec: BadgeSpec } | null = null;

    // Pass 1 keeps clear of other labels AND other spools' boxes, so no
    // spool gets hidden behind a neighbour's label. Pass 2 only avoids labels.
    for (const avoidBoxes of [true, false]) {
      for (const spec of req.variants) {
        const { w, h } = measureBadge(ctx, spec, font);
        for (const [x, y] of labelSpots(a, w, h)) {
          const rect = clamp(x, y, w, h);
          const blocked =
            placed.some((p) => overlaps(p.rect, rect)) ||
            (avoidBoxes && obstacles.some((o) => o !== a && overlaps(o, rect, 0)));
          if (!blocked) {
            chosen = { rect, spec };
            break;
          }
        }
        if (chosen) break;
      }
      if (chosen) break;
    }

    // No free spot: use the most compact wording at the position that
    // covers the least of the labels already placed.
    if (!chosen && req.force) {
      const spec = req.variants[req.variants.length - 1];
      const { w, h } = measureBadge(ctx, spec, font);
      let leastCovered = Infinity;
      for (const [x, y] of labelSpots(a, w, h)) {
        const rect = clamp(x, y, w, h);
        const covered = placed.reduce((sum, p) => sum + overlapArea(p.rect, rect), 0);
        if (covered < leastCovered) {
          leastCovered = covered;
          chosen = { rect, spec };
        }
      }
    }
    if (chosen) placed.push(chosen);
  }
  return placed;
}

/**
 * Draw one detection box in its band color. The line also gets thicker with
 * every band (3 px at 0–9 % up to 12 px at 90–100 %), so a closer match
 * stands out even for someone who can't tell the colors apart. An outline in
 * black (light colors) or white (dark colors) keeps it visible on any background.
 */
function drawDetectionBox(ctx: CanvasRenderingContext2D, d: Detection, r: ScreenRect): void {
  const color = MATCH_BAND_COLORS[d.band];
  const width = 3 + d.band;
  const outline = readableTextColor(color) === COLORS.black ? COLORS.black : COLORS.white;

  const radius = Math.min(16, r.w / 4, r.h / 4);

  ctx.setLineDash([]);
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.roundRect(r.x, r.y, r.w, r.h, radius);
  ctx.strokeStyle = outline;
  ctx.lineWidth = width + 4;
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

const CLOTH_RING_RADIUS = 26;

/** Draw the cloth target: a black/white double ring (visible on any color) with a center dot. */
function drawClothRing(ctx: CanvasRenderingContext2D, cx: number, cy: number): void {
  ctx.setLineDash([]);
  ctx.lineWidth = 9;
  ctx.strokeStyle = COLORS.black;
  ctx.beginPath();
  ctx.arc(cx, cy, CLOTH_RING_RADIUS, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 4;
  ctx.strokeStyle = COLORS.white;
  ctx.stroke();

  ctx.fillStyle = COLORS.white;
  ctx.strokeStyle = COLORS.black;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

/**
 * Paint the whole overlay: boxes and ring first, then labels. Label
 * priority is the closest spool › CLOTH › the other spools by score. Labels
 * are painted in reverse priority, so the most important one ends up on top.
 */
function paintOverlay(
  ctx: CanvasRenderingContext2D,
  items: { d: Detection; rect: ScreenRect }[],
  cloth: { x: number; y: number } | null,
  bounds: DrawBounds,
  font: string,
): void {
  for (const { d, rect } of items) drawDetectionBox(ctx, d, rect);
  if (cloth) drawClothRing(ctx, cloth.x, cloth.y);

  const requests: LabelRequest[] = [...items]
    .sort((a, b) => b.d.score - a.d.score)
    .map(({ d, rect }) => ({ anchor: rect, variants: badgeVariants(d), force: true }));

  if (cloth) {
    const r = CLOTH_RING_RADIUS + 4;
    const clothLabel: LabelRequest = {
      anchor: { x: cloth.x - r, y: cloth.y - r, w: r * 2, h: r * 2 },
      variants: [{ lines: ["CLOTH"], style: CLOTH_BADGE_STYLE }],
      force: false,
    };
    // Right after the closest spool (if any).
    requests.splice(Math.min(1, requests.length), 0, clothLabel);
  }

  const labels = layoutLabels(ctx, requests, items.map((i) => i.rect), bounds, font);
  for (let i = labels.length - 1; i >= 0; i--) paintBadge(ctx, labels[i].rect, labels[i].spec, font);
}

/* -------------------------------------------------------------------------- */
/*                                 Component                                  */
/* -------------------------------------------------------------------------- */

export default function CameraScanner() {
  // --- DOM refs ---------------------------------------------------------
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  /** Offscreen canvas for reading downscaled frame pixels. */
  const procCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // --- Mutable state read by the processing loop (refs avoid re-renders) -
  const streamRef = useRef<MediaStream | null>(null);
  const cvRef = useRef<OpenCV | null>(null);
  const clothRef = useRef<ClothTarget | null>(null);
  /** A tap waiting to be sampled on the next frame (normalised video coords). */
  const pendingTapRef = useRef<{ u: number; v: number } | null>(null);
  const voiceRef = useRef(false);
  const announcerRef = useRef<AnnouncerState>(freshAnnouncer());
  const rippleIdRef = useRef(0);
  const fontFamilyRef = useRef("system-ui, sans-serif");

  // --- React state (drives the visible UI) -------------------------------
  const [camera, setCamera] = useState<CameraState>({ kind: "starting" });
  const [cameraAttempt, setCameraAttempt] = useState(0);
  const [engine, setEngine] = useState<Engine>("loading");
  const [torchSupported, setTorchSupported] = useState(false);
  const [torchOn, setTorchOn] = useState(false);
  const [voiceOn, setVoiceOn] = useState<boolean>(loadVoicePreference);
  const [clothCss, setClothCss] = useState<string | null>(null);
  const [ripples, setRipples] = useState<Ripple[]>([]);
  const [summary, setSummary] = useState<Omit<Summary, "speech" | "key">>(() => {
    const s = summarize([], false);
    return { level: s.level, band: s.band, spool: s.spool, pct: s.pct, text: s.text };
  });

  useEffect(() => {
    voiceRef.current = voiceOn;
  }, [voiceOn]);

  // The overlay canvas uses the same accessible font as the rest of the UI.
  useEffect(() => {
    fontFamilyRef.current = getComputedStyle(document.body).fontFamily || fontFamilyRef.current;
  }, []);

  /* ------------------------------ Camera ------------------------------ */

  // (Re)open the camera on mount, on TRY AGAIN, and when the app becomes visible again.
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;

    (async () => {
      try {
        stream = await openCamera();
        if (cancelled) {
          stopStream(stream);
          return;
        }
        streamRef.current = stream;

        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          // Autoplay of a muted inline video is allowed; ignore the rare rejection.
          await video.play().catch(() => undefined);
        }

        const track = stream.getVideoTracks()[0];
        const caps = track?.getCapabilities?.() as TorchCapabilities | undefined;
        setTorchSupported(Boolean(caps?.torch));
        setTorchOn(false);
        setCamera({ kind: "running" });
      } catch (err) {
        if (!cancelled) setCamera({ kind: "error", message: describeCameraError(err) });
      }
    })();

    return () => {
      cancelled = true;
      stopStream(stream);
      if (streamRef.current === stream) streamRef.current = null;
    };
  }, [cameraAttempt]);

  // Release the camera (and the torch) while the app is in the background.
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden) {
        stopStream(streamRef.current);
        streamRef.current = null;
        setTorchOn(false);
      } else {
        setCamera({ kind: "starting" });
        setCameraAttempt((n) => n + 1);
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);

  const retryCamera = () => {
    setCamera({ kind: "starting" });
    setCameraAttempt((n) => n + 1);
  };

  /* ------------------------------ OpenCV ------------------------------ */

  // Switch from the canvas fallback to OpenCV as soon as the engine is ready.
  useEffect(() => {
    let alive = true;
    waitForOpenCv()
      .then((cv) => {
        if (!alive) return;
        cvRef.current = cv;
        setEngine("opencv");
      })
      .catch((err: unknown) => {
        console.warn("[ColorVision] OpenCV unavailable, using basic detector:", err);
        if (alive) setEngine("basic");
      });
    return () => {
      alive = false;
    };
  }, []);

  /* --------------------------- Announcements -------------------------- */

  /**
   * Speak + vibrate only when a result is *stable* (same for
   * ANNOUNCE_STABLE_FRAMES frames), differs from the last announcement, and
   * the cooldown has passed. This keeps feedback calm instead of chattering
   * every 300 ms while the camera wobbles.
   */
  const announce = useCallback((s: Summary) => {
    const a = announcerRef.current;
    if (s.key === a.candidate) a.streak++;
    else {
      a.candidate = s.key;
      a.streak = 1;
    }
    if (a.streak < ANNOUNCE_STABLE_FRAMES || s.key === a.announced) return;

    // States with nothing to say just get recorded, so a later match is "new" again.
    if (!s.speech) {
      a.announced = s.key;
      return;
    }

    const now = performance.now();
    if (now - a.lastAt < ANNOUNCE_COOLDOWN_MS) return;
    a.announced = s.key;
    a.lastAt = now;

    if (voiceRef.current) speak(s.speech);
    if (s.level === "best") vibrate(HAPTIC_BEST_MATCH);
    else if (s.level === "none") vibrate(HAPTIC_NO_MATCH);
  }, []);

  /* ---------------------------- Processing ---------------------------- */

  /** Clear and redraw the overlay for the current frame. */
  const drawOverlay = useCallback(
    (detections: Detection[], vw: number, vh: number, pw: number) => {
      const canvas = overlayRef.current;
      if (!canvas) return;
      const cssW = canvas.clientWidth;
      const cssH = canvas.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      // Keep the backing store at device resolution so lines and text stay sharp.
      const bw = Math.round(cssW * dpr);
      const bh = Math.round(cssH * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      const t = fitTransform(cssW, cssH, vw, vh);
      const k = (vw / pw) * t.scale; // processing px → screen px
      // Keep the video's CSS fit in sync with the transform used for drawing.
      const video = videoRef.current;
      if (video && video.style.objectFit !== t.mode) video.style.objectFit = t.mode;

      // The camera area has no bars on top of it, so labels can use all of it.
      const bounds: DrawBounds = { width: cssW, top: 0, bottom: cssH };

      const items = detections.map((d) => ({
        d,
        rect: { x: d.x * k + t.dx, y: d.y * k + t.dy, w: d.width * k, h: d.height * k },
      }));
      const cloth = clothRef.current;
      const clothPoint = cloth
        ? { x: cloth.u * vw * t.scale + t.dx, y: cloth.v * vh * t.scale + t.dy }
        : null;

      paintOverlay(ctx, items, clothPoint, bounds, fontFamilyRef.current);
    },
    [],
  );

  /** Analyse one camera frame. Runs every PROCESS_INTERVAL_MS. */
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const pw = PROCESSING_WIDTH;
    const ph = Math.round((pw * vh) / vw);

    // 1. Grab a downscaled copy of the frame.
    let proc = procCanvasRef.current;
    if (!proc) {
      proc = document.createElement("canvas");
      procCanvasRef.current = proc;
    }
    if (proc.width !== pw || proc.height !== ph) {
      proc.width = pw;
      proc.height = ph;
    }
    const pctx = proc.getContext("2d", { willReadFrequently: true });
    if (!pctx) return;
    pctx.drawImage(video, 0, 0, pw, ph);
    const img = pctx.getImageData(0, 0, pw, ph);

    // 2. Lock the cloth color if the user just tapped.
    const tap = pendingTapRef.current;
    if (tap) {
      pendingTapRef.current = null;
      const lab = sampleClothLab(img, tap.u * pw, tap.v * ph);
      if (lab) {
        clothRef.current = { lab, ...tap };
        announcerRef.current = freshAnnouncer();
        setClothCss(labToCssRgb(lab));
        if (voiceRef.current) speak("Cloth color locked. Now show the threads.");
      }
    }

    // 3. Detect and score spools.
    const cloth = clothRef.current;
    let detections: Detection[] = [];
    if (cloth) {
      const exclude = { x: cloth.u * pw, y: cloth.v * ph };
      let regions: Region[];
      const cv = cvRef.current;
      if (cv) {
        try {
          regions = detectRegionsOpenCv(cv, img, exclude);
        } catch (err) {
          // e.g. a WebAssembly memory error: use the fallback for this frame.
          console.warn("[ColorVision] OpenCV frame failed, using fallback:", err);
          regions = detectRegionsCanvas(img, exclude);
        }
      } else {
        regions = detectRegionsCanvas(img, exclude);
      }
      detections = rankRegions(regions, cloth.lab);
    }

    // 4. Render the overlay, the banner and the feedback.
    drawOverlay(detections, vw, vh, pw);

    const s = summarize(detections, cloth !== null);
    setSummary((prev) =>
      prev.text === s.text && prev.band === s.band
        ? prev
        : { level: s.level, band: s.band, spool: s.spool, pct: s.pct, text: s.text },
    );
    announce(s);
  }, [announce, drawOverlay]);

  // Fixed-rate loop: a chained timeout (not setInterval) so a slow frame
  // never makes runs overlap or pile up.
  useEffect(() => {
    let timer = 0;
    let stopped = false;
    const tick = () => {
      if (stopped) return;
      try {
        processFrame();
      } catch (err) {
        console.error("[ColorVision] frame processing error:", err);
      }
      timer = window.setTimeout(tick, PROCESS_INTERVAL_MS);
    };
    timer = window.setTimeout(tick, PROCESS_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [processFrame]);

  /* --------------------------- User actions --------------------------- */

  /** Set the cloth target at a screen point (CSS px relative to the overlay). */
  const pickClothAt = useCallback((sx: number, sy: number) => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas || !video.videoWidth) return;

    const t = fitTransform(canvas.clientWidth, canvas.clientHeight, video.videoWidth, video.videoHeight);
    const u = (sx - t.dx) / t.scale / video.videoWidth;
    const v = (sy - t.dy) / t.scale / video.videoHeight;
    // A tap on the black bars around a "contain" frame isn't on the cloth.
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    pendingTapRef.current = { u, v };

    const id = ++rippleIdRef.current;
    setRipples((rs) => [...rs, { id, x: sx, y: sy }]);
  }, []);

  const onOverlayPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    pickClothAt(e.clientX - rect.left, e.clientY - rect.top);
  };

  // Keyboard / switch access: Enter or Space picks the center of the view.
  const onOverlayKeyDown = (e: KeyboardEvent<HTMLCanvasElement>) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    const c = e.currentTarget;
    pickClothAt(c.clientWidth / 2, c.clientHeight / 2);
  };

  const resetCloth = () => {
    clothRef.current = null;
    pendingTapRef.current = null;
    announcerRef.current = freshAnnouncer();
    setClothCss(null);
    if (voiceRef.current) speak("Cloth color cleared. Tap the cloth.");
  };

  const toggleVoice = () => {
    const next = !voiceOn;
    voiceRef.current = next;
    setVoiceOn(next);
    saveVoicePreference(next);
    // Speaking inside the tap also unlocks speech on iOS.
    if (next) speak("Voice guidance on.");
    else stopSpeaking();
  };

  const toggleTorch = async () => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track || !torchSupported) return;
    const next = !torchOn;
    try {
      await track.applyConstraints({ advanced: [{ torch: next } as TorchConstraintSet] });
      setTorchOn(next);
      if (voiceRef.current) speak(next ? "Flash light on." : "Flash light off.");
    } catch (err) {
      console.warn("[ColorVision] torch failed:", err);
      setTorchSupported(false);
      setTorchOn(false);
    }
  };

  /* ------------------------------ Render ------------------------------ */

  const locked = clothCss !== null;
  const resultColor = summary.band !== null ? MATCH_BAND_COLORS[summary.band] : null;
  const resultText = resultColor ? readableTextColor(resultColor) : null;

  return (
    <main
      className="flex h-dvh w-full touch-manipulation select-none flex-col bg-black text-white"
      data-engine={engine}
    >
      {/* ---------------- Top bar: status, flash, live result ---------------- */}
      <header className="flex flex-col gap-2 px-3 pb-2 pt-[max(0.5rem,env(safe-area-inset-top))]">
        <div className="flex gap-2">
          <div
            role="status"
            className={`flex min-h-16 flex-1 items-center gap-3 rounded-2xl px-4 text-xl font-bold leading-tight ring-2 ring-inset ${
              locked ? "bg-white/[0.08] text-white ring-white/25" : "bg-cv-yellow/10 text-cv-yellow ring-cv-yellow"
            }`}
          >
            {locked ? <LockIcon size={26} /> : <UnlockIcon size={26} />}
            <span className="flex-1">{locked ? "CLOTH LOCKED" : "TAP THE CLOTH"}</span>
            {locked && (
              // The swatch is extra information; the text and lock icon carry the status.
              <span
                className="h-10 w-10 shrink-0 rounded-xl ring-2 ring-white"
                style={{ backgroundColor: clothCss ?? undefined }}
                aria-hidden="true"
              />
            )}
          </div>

          <button
            type="button"
            onClick={toggleTorch}
            disabled={!torchSupported}
            aria-pressed={torchSupported ? torchOn : undefined}
            aria-label={torchSupported ? `Flash light ${torchOn ? "on" : "off"}` : "Flash light not available"}
            className={`flex min-h-16 min-w-[96px] flex-col items-center justify-center gap-0.5 rounded-2xl px-3 text-base font-bold leading-none transition active:scale-95 disabled:opacity-45 ${
              torchOn ? "bg-cv-yellow text-black" : "bg-white/[0.08] text-white ring-2 ring-inset ring-white/25"
            }`}
          >
            {torchOn ? <FlashlightIcon size={26} /> : <FlashlightOffIcon size={26} />}
            <span>FLASH {!torchSupported ? "N/A" : torchOn ? "ON" : "OFF"}</span>
          </button>
        </div>

        {/* Live result card, read out by screen readers when it changes. */}
        <div
          aria-live="polite"
          className="flex min-h-16 items-center gap-3 rounded-2xl bg-white/[0.08] p-2 pr-4 ring-1 ring-inset ring-white/15"
        >
          {resultColor && resultText ? (
            <>
              <span
                className="flex h-14 min-w-[84px] shrink-0 items-center justify-center rounded-xl px-2 text-[28px] font-bold"
                style={{ backgroundColor: resultColor, color: resultText }}
              >
                {summary.pct}%
              </span>
              <span className="flex flex-col leading-tight">
                <span className="flex items-center gap-1.5 text-base font-bold uppercase tracking-wide text-white/75">
                  <StarIcon size={16} /> Closest match
                </span>
                <span className="text-2xl font-bold">Thread {summary.spool}</span>
              </span>
            </>
          ) : (
            <>
              <span
                className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-xl ${
                  summary.level === "idle" ? "bg-cv-yellow text-black" : "bg-white/15 text-white"
                }`}
              >
                {summary.level === "idle" ? <TargetIcon size={30} /> : <ScanIcon size={30} />}
              </span>
              <span className="text-xl font-bold leading-tight">{summary.text}</span>
            </>
          )}
        </div>
      </header>

      {/* ---------------- Camera area: video + overlay, nothing on top ---------------- */}
      <section className="relative mx-2 min-h-0 flex-1 overflow-hidden rounded-[28px] bg-neutral-950 ring-1 ring-white/10">
        {/* Live camera feed (decorative for screen readers: results are announced in text). */}
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full object-cover"
          playsInline
          muted
          autoPlay
          aria-hidden="true"
        />

        {/* Overlay: detection boxes, badges and the cloth target. Tapping it picks the cloth. */}
        <canvas
          ref={overlayRef}
          className="absolute inset-0 h-full w-full cursor-crosshair"
          role="button"
          tabIndex={0}
          aria-label="Camera view. Tap on the cloth to lock its color."
          onPointerDown={onOverlayPointerDown}
          onKeyDown={onOverlayKeyDown}
        />

        {/* Tap ripples, removed when their animation ends. */}
        {ripples.map((r) => (
          <span
            key={r.id}
            className="cv-ripple"
            style={{ left: r.x, top: r.y }}
            aria-hidden="true"
            onAnimationEnd={() => setRipples((rs) => rs.filter((x) => x.id !== r.id))}
          />
        ))}
      </section>

      {/* ---------------- Bottom bar: color key, reset, voice ---------------- */}
      <footer className="flex flex-col gap-2 px-3 pt-2 pb-[max(0.5rem,env(safe-area-inset-bottom))]">
        {/* Color key for the ten 10% bands: dark = far from the cloth color, bright = close. */}
        <div
          className="flex items-center gap-2 text-base font-bold text-white/85"
          role="img"
          aria-label="Color key: dark purple means 0 percent match, bright yellow means 100 percent match."
        >
          <span>0%</span>
          <div className="flex h-4 flex-1 overflow-hidden rounded-full ring-1 ring-white/40">
            {MATCH_BAND_COLORS.map((color, band) => (
              <span key={color} className="flex-1" style={{ backgroundColor: color }} title={matchBandLabel(band)} />
            ))}
          </div>
          <span>100%</span>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={resetCloth}
            className="flex min-h-[76px] flex-[3] items-center justify-center gap-2 rounded-full bg-cv-yellow px-4 text-[22px] font-bold leading-tight text-black shadow-[0_6px_24px_rgba(255,215,0,0.25)] transition active:scale-[0.97]"
          >
            <ResetIcon size={28} className="shrink-0" />
            <span>RESET CLOTH COLOR</span>
          </button>
          <button
            type="button"
            onClick={toggleVoice}
            aria-pressed={voiceOn}
            className={`flex min-h-[76px] flex-[2] items-center justify-center gap-2 rounded-full px-3 font-bold leading-tight transition active:scale-[0.97] ${
              voiceOn ? "bg-white text-black" : "bg-white/[0.08] text-white ring-2 ring-inset ring-white/25"
            }`}
          >
            {voiceOn ? <VolumeOnIcon size={28} className="shrink-0" /> : <VolumeOffIcon size={28} className="shrink-0" />}
            <span className="flex flex-col items-start text-left">
              <span className="text-base">VOICE GUIDANCE</span>
              <span className="text-xl">{voiceOn ? "ON" : "OFF"}</span>
            </span>
          </button>
        </div>
      </footer>

      {/* ---------------- Camera starting / error screens ---------------- */}
      {camera.kind === "starting" && (
        <div className="fixed inset-0 z-20 flex flex-col items-center justify-center gap-5 bg-black p-6 text-center">
          <span className="flex h-24 w-24 items-center justify-center rounded-full bg-white/10 text-white">
            <CameraIcon size={48} />
          </span>
          <p className="text-3xl font-bold text-white" role="status">
            Starting camera…
          </p>
        </div>
      )}

      {camera.kind === "error" && (
        <div
          role="alertdialog"
          aria-labelledby="camera-error-title"
          aria-describedby="camera-error-text"
          className="fixed inset-0 z-20 flex items-center justify-center bg-black p-5"
        >
          <div className="flex w-full max-w-md flex-col items-center gap-6 rounded-[28px] bg-white/[0.06] p-6 text-center ring-1 ring-white/15">
            <span className="flex h-20 w-20 items-center justify-center rounded-full bg-cv-red text-white">
              <AlertIcon size={40} />
            </span>
            <p id="camera-error-title" className="text-3xl font-bold text-white">
              Camera problem
            </p>
            <p id="camera-error-text" className="text-xl font-bold leading-snug text-white/90">
              {camera.message}
            </p>
            <button
              type="button"
              onClick={retryCamera}
              autoFocus
              className="flex min-h-[76px] w-full items-center justify-center gap-3 rounded-full bg-cv-yellow text-2xl font-bold text-black active:scale-[0.97]"
            >
              <ResetIcon size={28} />
              TRY AGAIN
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
