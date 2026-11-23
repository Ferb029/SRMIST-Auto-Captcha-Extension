// Answers "warmup" / "solve" requests. Shared by the Chrome/Edge offscreen page and the Firefox
// background page, so the model is loaded once per page and reused for every captcha.

import { createSolver } from "./solver.js";

let ready = null;

export async function handle(message) {
  ready ??= createSolver();
  const solver = await ready;
  if (message.type === "warmup") return { ok: true };
  if (message.type === "solve") {
    const started = performance.now();
    const blob = await (await fetch(message.image)).blob();   // data: URL from the content script
    const text = await solver.solveBlob(blob);
    return { ok: true, text, ms: Math.round(performance.now() - started) };
  }
  return { ok: false, error: `unknown request: ${message.type}` };
}
