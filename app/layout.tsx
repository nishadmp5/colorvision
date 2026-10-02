import type { Metadata, Viewport } from "next";
import { Atkinson_Hyperlegible } from "next/font/google";
import Script from "next/script";
import ServiceWorkerRegistrar from "@/components/ServiceWorkerRegistrar";
import "./globals.css";

/**
 * Atkinson Hyperlegible was designed by the Braille Institute for low-vision
 * readers: very distinct letterforms (e.g. 1/l/I, 0/O). `next/font` downloads
 * it at build time and serves it from our own origin, so it works offline.
 */
const atkinson = Atkinson_Hyperlegible({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-atkinson",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ColorVision — Thread Matcher",
  description:
    "Point your camera at cloth and thread spools to find the best matching thread color. Works offline.",
  applicationName: "ColorVision",
  // PWA install metadata (public/manifest.json) and home-screen icons.
  manifest: "/manifest.json",
  icons: {
    icon: [
      { url: "/icons/favicon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
    ],
    apple: [{ url: "/icons/apple-touch-icon.png", sizes: "180x180" }],
  },
  // iOS "Add to Home Screen": run full-screen with a black status bar.
  appleWebApp: {
    capable: true,
    title: "ColorVision",
    statusBarStyle: "black",
  },
  // Stop iOS from turning numbers like "95%" into phone-number links.
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Draw edge-to-edge on notched phones; components use safe-area insets.
  viewportFit: "cover",
  themeColor: "#000000",
  colorScheme: "dark",
  // Pinch-zoom stays enabled for accessibility (WCAG 1.4.4).
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${atkinson.variable} h-full antialiased`}>
      <body className="h-full overflow-hidden">
        {children}
        {/*
          OpenCV.js (~10 MB, served from our own origin and precached by the
          service worker). `lazyOnload` waits for browser idle time, so the
          camera starts first. Scripts are always client-only. The scanner
          picks the module up through `waitForOpenCv()` in utils/opencv.ts.
        */}
        <Script id="opencv-js" src="/opencv.js" strategy="lazyOnload" />
        {/* Offline support: precaches the whole app (production builds only). */}
        <ServiceWorkerRegistrar />
      </body>
    </html>
  );
}
