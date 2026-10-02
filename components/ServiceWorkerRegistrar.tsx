"use client";

import { useEffect } from "react";

/**
 * Registers the offline service worker (`/sw.js`) in production builds.
 *
 * It's skipped in development: a caching worker would serve stale
 * hot-reload bundles. Registration waits for the `load` event, so the
 * ~10 MB precache download doesn't compete with the first camera frame.
 */
export default function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;

    const register = () => {
      navigator.serviceWorker
        .register("/sw.js", { scope: "/", updateViaCache: "none" })
        .catch((err: unknown) => console.warn("[ColorVision] service worker registration failed:", err));
    };

    if (document.readyState === "complete") register();
    else window.addEventListener("load", register, { once: true });
  }, []);

  return null;
}
