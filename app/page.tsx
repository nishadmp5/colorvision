"use client";

/**
 * Home page: loads the camera scanner on the client only.
 *
 * In Next 16, `next/dynamic` with `ssr: false` is only allowed in Client
 * Components, so this page is marked "use client". The scanner needs
 * `window`, the camera and canvas, so it must never be prerendered. The
 * static export only contains the loading screen below, which also shows
 * while the scanner chunk downloads.
 */
import dynamic from "next/dynamic";

function LoadingScreen() {
  return (
    <main className="flex h-dvh flex-col items-center justify-center gap-6 bg-black p-6 text-center">
      <h1 className="text-5xl font-bold text-cv-yellow">ColorVision</h1>
      <p className="text-3xl font-bold text-white" role="status">
        📷 STARTING CAMERA…
      </p>
    </main>
  );
}

const CameraScanner = dynamic(() => import("@/components/CameraScanner"), {
  ssr: false,
  loading: LoadingScreen,
});

export default function Home() {
  return <CameraScanner />;
}
