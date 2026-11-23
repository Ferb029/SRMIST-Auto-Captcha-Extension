// SRM student portal login page: reads the captcha image the page loaded and types the
// answer into the Captcha box. You still enter your password and click Login yourself.

(() => {
  "use strict";

  const ask = (type, image) => chrome.runtime.sendMessage({ target: "background", type, image });

  ask("warmup").catch(() => {});                // load the model while the page fetches its captcha

  function start() {
    const image = document.getElementById("secure_captcha");
    const input = document.getElementById("captcha");
    if (!image || !input) return;               // not the login form
    let solvedSrc = "";

    async function solve() {
      const src = image.getAttribute("src") || "";
      // The page loads the real captcha as a blob: image; anything else (e.g. its decoy) is ignored.
      if (!src.startsWith("blob:") || src === solvedSrc) return;
      solvedSrc = src;
      try {
        if (!image.complete || !image.naturalWidth) await image.decode();
        const result = await ask("solve", await toDataUrl(image));
        if (!result?.ok || !result.text) throw new Error(result?.error || "no text read");
        if (image.getAttribute("src") !== src) return;                       // a newer captcha arrived
        if (input.value && input.dataset.srmCaptcha !== "filled") return;    // you already typed one
        fill(input, result.text, result.ms);
      } catch (error) {
        input.dataset.srmCaptcha = "failed";
        input.title = "SRMIST Auto Captcha could not read this captcha - please type it";
        console.warn("[SRMIST Auto Captcha]", error);
      }
    }

    new MutationObserver(solve).observe(image, { attributes: true, attributeFilter: ["src"] });
    image.addEventListener("load", solve);
    solve();
  }

  // Original PNG bytes when possible (exact pixels); a canvas copy otherwise.
  async function toDataUrl(image) {
    try {
      const blob = await (await fetch(image.src)).blob();
      return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
    } catch {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext("2d").drawImage(image, 0, 0);
      return canvas.toDataURL("image/png");
    }
  }

  function fill(input, text, ms) {
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.dataset.srmCaptcha = "filled";
    input.title = `Filled by SRMIST Auto Captcha (${ms} ms)`;
    input.animate(
      [{ boxShadow: "0 0 0 3px rgba(22, 163, 74, .55)" }, { boxShadow: "0 0 0 0 rgba(22, 163, 74, 0)" }],
      { duration: 1200, easing: "ease-out" },
    );
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
