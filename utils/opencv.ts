/**
 * OpenCV.js loading and typing.
 *
 * `public/opencv.js` is injected by `<Script strategy="lazyOnload">` in the
 * root layout, so the camera can start before the ~10 MB engine is parsed.
 * Until OpenCV is ready, the scanner uses its pure-canvas fallback detector.
 *
 * Depending on the build, OpenCV.js exposes `window.cv` as:
 *   1. a Promise that resolves to the ready module (MODULARIZE builds, 4.8+), or
 *   2. the Emscripten Module object, ready once `onRuntimeInitialized` fires.
 * `waitForOpenCv()` copes with both. Emscripten builds (including the pinned
 * 4.9.0) give the module a `then()` that resolves to the module itself, so
 * awaiting it or resolving a Promise with it loops forever. We never await
 * it, and we delete that `then` before handing the module out.
 */

/* -------------------------------------------------------------------------- */
/*            Minimal typings (only the API surface ColorVision uses)          */
/* -------------------------------------------------------------------------- */

export interface CvMat {
  rows: number;
  cols: number;
  data: Uint8Array;
  delete(): void;
  isDeleted(): boolean;
}

export interface CvMatVector {
  size(): number;
  get(index: number): CvMat;
  push_back(mat: CvMat): void;
  delete(): void;
}

export interface CvRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CvSize {
  width: number;
  height: number;
}

export interface CvPoint {
  x: number;
  y: number;
}

/** A 4-element scalar; `cv.mean()` returns one as a plain number array. */
export type CvScalarValue = [number, number, number, number];

export interface OpenCV {
  Mat: {
    new (): CvMat;
    new (rows: number, cols: number, type: number): CvMat;
    zeros(rows: number, cols: number, type: number): CvMat;
    ones(rows: number, cols: number, type: number): CvMat;
  };
  MatVector: new () => CvMatVector;
  Size: new (width: number, height: number) => CvSize;
  Point: new (x: number, y: number) => CvPoint;
  Scalar: new (v0: number, v1?: number, v2?: number, v3?: number) => CvScalarValue;

  matFromImageData(imageData: ImageData): CvMat;
  cvtColor(src: CvMat, dst: CvMat, code: number, dstCn?: number): void;
  medianBlur(src: CvMat, dst: CvMat, ksize: number): void;
  Canny(src: CvMat, edges: CvMat, threshold1: number, threshold2: number): void;
  getStructuringElement(shape: number, ksize: CvSize): CvMat;
  dilate(src: CvMat, dst: CvMat, kernel: CvMat, anchor?: CvPoint, iterations?: number): void;
  erode(src: CvMat, dst: CvMat, kernel: CvMat, anchor?: CvPoint, iterations?: number): void;
  morphologyEx(src: CvMat, dst: CvMat, op: number, kernel: CvMat): void;
  findContours(
    image: CvMat,
    contours: CvMatVector,
    hierarchy: CvMat,
    mode: number,
    method: number,
  ): void;
  contourArea(contour: CvMat): number;
  boundingRect(contour: CvMat): CvRect;
  convexHull(points: CvMat, hull: CvMat): void;
  drawContours(
    image: CvMat,
    contours: CvMatVector,
    contourIdx: number,
    color: CvScalarValue,
    thickness: number,
  ): void;
  mean(src: CvMat, mask?: CvMat): CvScalarValue;
  countNonZero(src: CvMat): number;
  split(src: CvMat, channels: CvMatVector): void;
  bitwise_or(src1: CvMat, src2: CvMat, dst: CvMat): void;

  COLOR_RGBA2GRAY: number;
  COLOR_RGBA2RGB: number;
  COLOR_RGB2Lab: number;
  CV_8UC1: number;
  MORPH_RECT: number;
  MORPH_CLOSE: number;
  RETR_EXTERNAL: number;
  CHAIN_APPROX_SIMPLE: number;
}

/** Whatever OpenCV.js has put on `window.cv` so far. */
type CvGlobal =
  | (Partial<OpenCV> & {
      onRuntimeInitialized?: () => void;
      then?: (cb: (mod: OpenCV) => void) => unknown;
    })
  | undefined;

declare global {
  interface Window {
    cv?: CvGlobal;
  }
}

/* -------------------------------------------------------------------------- */
/*                                   Loader                                   */
/* -------------------------------------------------------------------------- */

/** True when the object is a fully initialised OpenCV module. */
function isReady(mod: unknown): mod is OpenCV {
  return (
    typeof mod === "object" &&
    mod !== null &&
    typeof (mod as Partial<OpenCV>).Mat === "function" &&
    typeof (mod as Partial<OpenCV>).findContours === "function"
  );
}

let readyPromise: Promise<OpenCV> | null = null;

/**
 * Resolve with the initialised OpenCV module once it's usable.
 * The same promise is shared between callers.
 *
 * @param timeoutMs Reject after this long (slow phones can take several
 *                  seconds to compile the WebAssembly; 45 s is generous).
 */
export function waitForOpenCv(timeoutMs = 45_000): Promise<OpenCV> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("OpenCV.js is browser-only"));
  }
  if (readyPromise) return readyPromise;

  readyPromise = new Promise<OpenCV>((resolve, reject) => {
    let settled = false;
    let hooked: CvGlobal = undefined;

    const finish = (mod: OpenCV) => {
      if (settled) return;
      settled = true;
      clearInterval(poll);
      clearTimeout(timer);
      // Strip the self-resolving `then()`. Otherwise `resolve(mod)` would
      // treat the module as a thenable and unwrap it forever, freezing the
      // main thread.
      delete (mod as { then?: unknown }).then;
      window.cv = mod; // Normalise: from now on window.cv is the ready module.
      resolve(mod);
    };

    const check = () => {
      const cv = window.cv;
      if (!cv) return; // Script not executed yet.
      if (isReady(cv)) return finish(cv);
      if (cv === hooked) return; // Already waiting on this object.
      hooked = cv;

      if (typeof cv.then === "function") {
        // Promise-style build: register a callback once, never await it.
        cv.then((mod) => {
          if (isReady(mod)) finish(mod);
        });
      } else {
        // Module-object build: chain onto the runtime-initialised hook.
        const previous = cv.onRuntimeInitialized;
        cv.onRuntimeInitialized = () => {
          previous?.();
          if (isReady(window.cv)) finish(window.cv);
        };
      }
    };

    // Poll cheaply: the <Script> is lazy-loaded, so window.cv shows up late.
    const poll = setInterval(check, 150);
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      clearInterval(poll);
      readyPromise = null; // Allow a later retry.
      reject(new Error("OpenCV.js did not initialise in time"));
    }, timeoutMs);
    check();
  });

  return readyPromise;
}
