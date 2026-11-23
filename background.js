// Background: receives solve requests from the login page and runs the OCR model.
//
// Chrome / Edge: the background is a service worker, which is stopped when idle and can't keep the
//   model loaded, so the model lives in a hidden offscreen page (chrome.offscreen) that stays
//   alive; requests are relayed to it. The model is loaded once.
// Firefox: has no offscreen API, but its background is a real page that can run the model itself.

const hasOffscreen = typeof chrome.offscreen?.createDocument === "function";
let creating = null;
let localSolver = null;

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  creating ??= chrome.offscreen
    .createDocument({
      url: "offscreen.html",
      reasons: [chrome.offscreen.Reason.BLOBS],
      justification: "Decode the captcha image and run the OCR model on it.",
    })
    .finally(() => { creating = null; });
  await creating;
}

async function handle(message) {
  if (hasOffscreen) {
    await ensureOffscreen();
    return chrome.runtime.sendMessage({ ...message, target: "offscreen" });
  }
  localSolver ??= import("./solver-service.js");
  return (await localSolver).handle(message);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "background") return;
  handle(message).then(sendResponse, (error) => sendResponse({ ok: false, error: String(error?.message || error) }));
  return true;                                  // respond asynchronously
});
