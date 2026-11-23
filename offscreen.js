// Chrome / Edge: hidden extension page that keeps the OCR model loaded and answers solve requests.

import { handle } from "./solver-service.js";

handle({ type: "warmup" }).catch(() => {});     // start loading the model right away

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen") return;
  handle(message).then(sendResponse, (error) => sendResponse({ ok: false, error: String(error?.message || error) }));
  return true;                                  // respond asynchronously
});
