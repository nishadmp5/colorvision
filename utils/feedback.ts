/**
 * Multisensory feedback: speech (Web Speech API) and haptics (Vibration API).
 *
 * Both APIs are feature-detected. Support varies:
 *   - speechSynthesis: all modern browsers. iOS only allows it after a user
 *     gesture, so the first utterance must come from a tap (the voice toggle).
 *   - navigator.vibrate: Android Chrome/Firefox. Not available on iOS Safari,
 *     where calls silently do nothing.
 */

/** Haptic pattern for "best match found": two short pulses. */
export const HAPTIC_BEST_MATCH: readonly number[] = [100, 50, 100];

/** Haptic pattern for "no match": one long buzz. */
export const HAPTIC_NO_MATCH: readonly number[] = [300];

/** Slightly slower than default speech, easier to follow for older listeners. */
const SPEECH_RATE = 0.9;

export function canSpeak(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/**
 * Speak `text` aloud. Any utterance still playing is cancelled first, so
 * the user always hears the newest result and messages never queue up.
 */
export function speak(text: string): void {
  if (!canSpeak()) return;
  const synth = window.speechSynthesis;
  synth.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = document.documentElement.lang || "en";
  utterance.rate = SPEECH_RATE;
  utterance.pitch = 1;
  utterance.volume = 1;
  synth.speak(utterance);
}

/** Stop any speech in progress (e.g. when voice guidance is switched off). */
export function stopSpeaking(): void {
  if (canSpeak()) window.speechSynthesis.cancel();
}

/** Vibrate with the given pattern; silently ignored when unsupported. */
export function vibrate(pattern: readonly number[]): void {
  if (typeof navigator === "undefined" || typeof navigator.vibrate !== "function") return;
  try {
    navigator.vibrate([...pattern]);
  } catch {
    // Some browsers throw when vibration is blocked (e.g. no user activation yet).
  }
}
