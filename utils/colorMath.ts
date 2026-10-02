/**
 * Color math engine: pure, dependency-free and side-effect-free.
 *
 * Pipeline used by the scanner:
 *
 *   sRGB (0–255) ──► linear RGB ──► CIE XYZ (D65) ──► CIELAB (L*a*b*)
 *                                                        │
 *   cloth Lab ─────────────── CIEDE2000 (ΔE₀₀) ◄─────────┘ spool Lab
 *                                   │
 *                     score = 100 · e^(−0.1 · ΔE₀₀)  ──► BEST / GOOD / NONE
 *
 * Why CIELAB + ΔE₀₀? RGB distance doesn't match human perception: two
 * colors with the same RGB distance can look nearly identical or clearly
 * different. CIELAB is (approximately) perceptually uniform, and CIEDE2000
 * fixes its remaining non-uniformities (blue region, neutrals, chroma/hue
 * weighting). It is the industry standard for textile color matching.
 *
 * References:
 *  - IEC 61966-2-1 (sRGB transfer function and primaries)
 *  - G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 Color-Difference Formula:
 *    Implementation Notes, Supplementary Test Data, and Mathematical
 *    Observations", Color Research & Application, 2005.
 */

/** An sRGB color with 8-bit channels (0–255). */
export interface Rgb {
  r: number;
  g: number;
  b: number;
}

/** A CIE XYZ color, scaled so that the reference white has Y = 1. */
export interface Xyz {
  x: number;
  y: number;
  z: number;
}

/** A CIELAB color. L ∈ [0, 100]; a and b are roughly within ±128. */
export interface Lab {
  L: number;
  a: number;
  b: number;
}

/** Match categories shown to the user (always with icon + text, never color alone). */
export type MatchLevel = "best" | "good" | "none";

/* -------------------------------------------------------------------------- */
/*                                 Constants                                  */
/* -------------------------------------------------------------------------- */

/** CIE standard illuminant D65 reference white (2° observer), Y normalised to 1. */
export const D65_WHITE: Readonly<Xyz> = { x: 0.95047, y: 1.0, z: 1.08883 };

/** Exact CIE constants for the Lab companding function (ε = 216/24389, κ = 24389/27). */
const LAB_EPSILON = 216 / 24389;
const LAB_KAPPA = 24389 / 27;

/** 25^7, used by CIEDE2000's chroma terms. */
const POW25_7 = 25 ** 7;

/** Minimum score (%) for "BEST MATCH": ΔE₀₀ ≤ ~2.9, a barely noticeable difference. */
export const BEST_MATCH_MIN_SCORE = 75;

/** Minimum score (%) for a match (✓): ΔE₀₀ ≤ ~6.9, a small, acceptable difference for thread. */
export const GOOD_MATCH_MIN_SCORE = 50;

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */

const toRadians = (deg: number): number => (deg * Math.PI) / 180;
const toDegrees = (rad: number): number => (rad * 180) / Math.PI;

/* -------------------------------------------------------------------------- */
/*                             RGB → XYZ → CIELAB                             */
/* -------------------------------------------------------------------------- */

/**
 * Undo the sRGB gamma curve: an 8-bit channel value (0–255) becomes linear
 * light intensity (0–1).
 */
export function srgbToLinear(channel: number): number {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/**
 * Convert 8-bit sRGB to CIE XYZ (D65), using the standard sRGB primaries
 * matrix from IEC 61966-2-1.
 */
export function rgbToXyz({ r, g, b }: Rgb): Xyz {
  const rl = srgbToLinear(r);
  const gl = srgbToLinear(g);
  const bl = srgbToLinear(b);

  return {
    x: 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl,
    y: 0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl,
    z: 0.0193339 * rl + 0.119192 * gl + 0.9503041 * bl,
  };
}

/**
 * Convert CIE XYZ to CIELAB relative to a reference white (D65 by default).
 */
export function xyzToLab({ x, y, z }: Xyz, white: Readonly<Xyz> = D65_WHITE): Lab {
  // Non-linear companding: cube root above ε, linear segment below it.
  const f = (t: number): number =>
    t > LAB_EPSILON ? Math.cbrt(t) : (LAB_KAPPA * t + 16) / 116;

  const fx = f(x / white.x);
  const fy = f(y / white.y);
  const fz = f(z / white.z);

  return {
    L: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz),
  };
}

/** Convenience: 8-bit sRGB straight to CIELAB (D65). */
export function rgbToLab(rgb: Rgb): Lab {
  return xyzToLab(rgbToXyz(rgb));
}

/* -------------------------------------------------------------------------- */
/*                            CIEDE2000 (ΔE₀₀)                                */
/* -------------------------------------------------------------------------- */

/**
 * CIEDE2000 color difference between two CIELAB colors.
 *
 * This follows the reference implementation notes by Sharma et al. (2005)
 * step by step, including the hue-angle edge cases. The test-data pairs from
 * that paper reproduce to 4 decimal places.
 *
 * kL, kC and kH are the parametric weighting factors; 1 is the standard
 * reference condition used for most applications, textiles included.
 *
 * @returns ΔE₀₀ ≥ 0. Rule of thumb: < 1 imperceptible, 1–3 barely visible,
 *          3–6 noticeable, > 10 clearly different colors.
 */
export function deltaE2000(lab1: Lab, lab2: Lab, kL = 1, kC = 1, kH = 1): number {
  const { L: L1, a: a1, b: b1 } = lab1;
  const { L: L2, a: a2, b: b2 } = lab2;

  // --- Step 1: compute a′, C′ and h′ ------------------------------------
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const Cbar = (C1 + C2) / 2;
  const Cbar7 = Cbar ** 7;
  // G compensates for the poor uniformity of CIELAB near the neutral axis.
  const G = 0.5 * (1 - Math.sqrt(Cbar7 / (Cbar7 + POW25_7)));

  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);

  // Hue angle in degrees within [0, 360); defined as 0 for achromatic colors.
  const hueAngle = (b: number, ap: number): number => {
    if (b === 0 && ap === 0) return 0;
    const h = toDegrees(Math.atan2(b, ap));
    return h >= 0 ? h : h + 360;
  };
  const h1p = hueAngle(b1, a1p);
  const h2p = hueAngle(b2, a2p);

  // --- Step 2: compute ΔL′, ΔC′ and ΔH′ ----------------------------------
  const dLp = L2 - L1;
  const dCp = C2p - C1p;

  const C1pC2p = C1p * C2p;
  let dhp = 0;
  if (C1pC2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1pC2p) * Math.sin(toRadians(dhp / 2));

  // --- Step 3: compute the weighting functions and the final ΔE₀₀ -------
  const Lbarp = (L1 + L2) / 2;
  const Cbarp = (C1p + C2p) / 2;

  // Mean hue, taking the wrap-around at 0°/360° into account.
  let hbarp = h1p + h2p;
  if (C1pC2p !== 0) {
    if (Math.abs(h1p - h2p) <= 180) hbarp /= 2;
    else if (h1p + h2p < 360) hbarp = (hbarp + 360) / 2;
    else hbarp = (hbarp - 360) / 2;
  }

  const T =
    1 -
    0.17 * Math.cos(toRadians(hbarp - 30)) +
    0.24 * Math.cos(toRadians(2 * hbarp)) +
    0.32 * Math.cos(toRadians(3 * hbarp + 6)) -
    0.2 * Math.cos(toRadians(4 * hbarp - 63));

  const dTheta = 30 * Math.exp(-(((hbarp - 275) / 25) ** 2));
  const Cbarp7 = Cbarp ** 7;
  const RC = 2 * Math.sqrt(Cbarp7 / (Cbarp7 + POW25_7));
  const Lbarp50sq = (Lbarp - 50) ** 2;
  const SL = 1 + (0.015 * Lbarp50sq) / Math.sqrt(20 + Lbarp50sq);
  const SC = 1 + 0.045 * Cbarp;
  const SH = 1 + 0.015 * Cbarp * T;
  // Rotation term: fixes the tilted ellipses in the blue region.
  const RT = -Math.sin(toRadians(2 * dTheta)) * RC;

  const lTerm = dLp / (kL * SL);
  const cTerm = dCp / (kC * SC);
  const hTerm = dHp / (kH * SH);

  return Math.sqrt(lTerm ** 2 + cTerm ** 2 + hTerm ** 2 + RT * cTerm * hTerm);
}

/* -------------------------------------------------------------------------- */
/*                             Scoring & categories                           */
/* -------------------------------------------------------------------------- */

/**
 * Turn a ΔE₀₀ distance into an easy-to-say percentage:
 *   score = 100 · e^(−0.1 · ΔE₀₀)
 * ΔE 0 → 100 %, ΔE 1 → 90 %, ΔE 3 → 74 %, ΔE 6.9 → 50 %, ΔE 10 → 37 %.
 */
export function matchScore(deltaE: number): number {
  return 100 * Math.exp(-0.1 * Math.max(0, deltaE));
}

/**
 * Classify a score (%) into the three user-facing match levels.
 * It uses the rounded score (the number shown on screen and spoken), so a
 * spool labelled "50%" is always a match, never ✗.
 */
export function classifyMatch(score: number): MatchLevel {
  const shown = Math.round(score);
  if (shown >= BEST_MATCH_MIN_SCORE) return "best";
  if (shown >= GOOD_MATCH_MIN_SCORE) return "good";
  return "none";
}

/* -------------------------------------------------------------------------- */
/*                                 Match bands                                */
/* -------------------------------------------------------------------------- */

/**
 * Indicator colors for the ten 10%-wide match bands: 0–9 %, 10–19 %, …,
 * 90–100 %. They follow the "viridis" scale (dark purple → blue → green →
 * bright yellow), which is designed to stay readable with color blindness:
 * its lightness rises steadily, so a brighter indicator always means a
 * closer match, even when the hues can't be told apart.
 */
export const MATCH_BAND_COLORS = [
  "#440154", // 0–9 %
  "#482878", // 10–19 %
  "#3E4989", // 20–29 %
  "#31688E", // 30–39 %
  "#26828E", // 40–49 %
  "#1F9E89", // 50–59 %
  "#35B779", // 60–69 %
  "#6ECE58", // 70–79 %
  "#B5DE2B", // 80–89 %
  "#FDE725", // 90–100 %
] as const;

/**
 * Band index 0–9 for a score: 0 = 0–9 %, … 9 = 90–100 %. It uses the rounded
 * score (the number shown on screen), so "90%" is always in the top band.
 */
export function matchBand(score: number): number {
  return Math.min(9, Math.max(0, Math.floor(Math.round(score) / 10)));
}

/** Human-readable range of a band, e.g. "70–79%" or "90–100%". */
export function matchBandLabel(band: number): string {
  return band >= 9 ? "90–100%" : `${band * 10}–${band * 10 + 9}%`;
}

/** Black or white, whichever has more contrast on the given `#RRGGBB` color. */
export function readableTextColor(hex: string): "#000000" | "#FFFFFF" {
  const n = parseInt(hex.slice(1), 16);
  const lum =
    0.2126 * srgbToLinear((n >> 16) & 255) +
    0.7152 * srgbToLinear((n >> 8) & 255) +
    0.0722 * srgbToLinear(n & 255);
  // Equal contrast against black and white happens at luminance ≈ 0.179.
  return lum > 0.179 ? "#000000" : "#FFFFFF";
}

/** Format a Lab color as a CSS `rgb()` string (used for the cloth swatch). */
export function labToCssRgb({ L, a, b }: Lab, white: Readonly<Xyz> = D65_WHITE): string {
  // Inverse of xyzToLab.
  const fy = (L + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;
  const finv = (t: number): number =>
    t ** 3 > LAB_EPSILON ? t ** 3 : (116 * t - 16) / LAB_KAPPA;
  const x = finv(fx) * white.x;
  const y = finv(fy) * white.y;
  const z = finv(fz) * white.z;

  // Inverse of the sRGB primaries matrix, then gamma encoding.
  const lin = [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
  const [r, g, bl] = lin.map((c) => {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  });
  return `rgb(${r}, ${g}, ${bl})`;
}
