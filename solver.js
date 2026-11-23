// Captcha OCR: ddddocr's model run with onnxruntime-web, fully offline.
// Image preparation reproduces Pillow exactly (Lanczos resize to height 64, then grayscale),
// so results match the Python ddddocr pipeline.

import * as ort from "./lib/ort.wasm.bundle.min.mjs";

const BASE = new URL(".", import.meta.url).href;
const HEIGHT = 64;
// Model output i (i >= 1) is ALPHABET[i - 1]; 0 is the CTC blank. Must match tools/extract_model.py.
const ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

export async function createSolver() {
  ort.env.wasm.numThreads = 1;                 // no cross-origin isolation in extension pages
  ort.env.wasm.wasmPaths = { wasm: BASE + "lib/ort-wasm-simd-threaded.wasm" };   // glue code is in the bundle

  const session = await ort.InferenceSession.create(BASE + "model/ocr.onnx", {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const inputName = session.inputNames[0];
  const groups = letterGroups(ALPHABET);

  async function solvePixels(rgba, width, height) {
    const pixels = preprocess(rgba, width, height);
    const input = new ort.Tensor("float32", pixels.data, [1, 1, HEIGHT, pixels.width]);
    const output = (await session.run({ [inputName]: input }))[session.outputNames[0]];
    return fixCase(decode(output, groups), findGlyphs(rgba, width, height), height);
  }

  async function solveBlob(blob) {
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const text = await solvePixels(data, bitmap.width, bitmap.height);
    bitmap.close();
    return text;
  }

  // First run compiles the kernels; do it now so the first real captcha is fast.
  await solvePixels(new Uint8ClampedArray(175 * 45 * 4).fill(255), 175, 45);
  return { solvePixels, solveBlob };
}

// ---- CTC decoding ------------------------------------------------------------------------------
// Model classes: 0 = blank, i = alphabet[i - 1] (the model is trimmed to these, see tools/).
// The model is good at WHICH letter it sees but poor at its case (it leans heavily to lowercase),
// so "k" and "K" are decoded as one letter (probabilities added) and the case is decided
// afterwards from the letter's size and shape in the image - see fixCase.

function letterGroups(alphabet) {
  const byKey = new Map();
  [...alphabet].forEach((char, i) => {
    const key = char.toLowerCase();
    if (!byKey.has(key)) byKey.set(key, { key, lower: -1, upper: -1 });
    const group = byKey.get(key);
    if (char === key) group.lower = i + 1; else group.upper = i + 1;
  });
  return [...byKey.values()];
}

// Returns one entry per character: { key, lower, upper } with the lower/upper-case logits taken
// from the time step where the model was most confident about that character.
function decode(output, groups) {
  const [steps, , classes] = output.dims;      // [time, batch=1, classes]
  const scores = output.data;
  const chars = [];
  let previous = -1;                            // -1 = blank
  for (let t = 0, row = 0; t < steps; t++, row += classes) {
    let best = -1, bestScore = scores[row];
    for (let g = 0; g < groups.length; g++) {
      const { lower, upper } = groups[g];
      const a = scores[row + lower];
      const score = upper < 0 ? a : logAddExp(a, scores[row + upper]);
      if (score > bestScore) { bestScore = score; best = g; }
    }
    if (best !== -1) {
      const { key, lower, upper } = groups[best];
      const entry = { key, score: bestScore, lower: scores[row + lower], upper: upper < 0 ? -Infinity : scores[row + upper] };
      if (best !== previous) chars.push(entry);
      else if (bestScore > chars[chars.length - 1].score) chars[chars.length - 1] = entry;
    }
    previous = best;
  }
  return chars;
}

const logAddExp = (a, b) => Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

// ---- Case from the image ------------------------------------------------------------------------
// SRM captchas draw each character in its own solid dark colour, so characters can be cut out by
// colour (thin noise lines are removed with a 2x2 opening). Rules measured on 89 labelled captchas
// (heights are for the 45 px tall image):
//   a c e m n o r s u v w x z  lowercase is short (16-19 px), uppercase is tall (21-26 px)
//   i j                        lowercase has a separate dot (an empty row inside the letter)
//   b d h k                    lowercase has only a thin stem in its top quarter
//   l                          L has a wide foot, l does not
//   q                          Q (with its tail) is 25-28 px, q is 21-24 px
//   f g p t y                  shape of the top / middle / bottom (see below)
//   anything else / unsure     fall back to the model, corrected for its lowercase bias

const SHORT_LOWER = "acemnorsuvwxz";
const MODEL_LOWERCASE_BIAS = 3.7;               // logit margin the model gives lowercase by habit

function fixCase(chars, glyphs, imageHeight) {
  const aligned = glyphs.length === chars.length;
  const scale = imageHeight / 45;
  return chars.map(({ key, lower, upper }, i) => {
    if (upper === -Infinity) return key;        // digit
    const fromImage = aligned ? caseFromShape(key, glyphs[i], scale) : null;
    const isUpper = fromImage ?? lower - upper < MODEL_LOWERCASE_BIAS;
    return isUpper ? key.toUpperCase() : key;
  }).join("");
}

// true = uppercase, false = lowercase, null = can't tell from the shape
function caseFromShape(key, g, scale) {
  const h = g.height / scale;
  if (SHORT_LOWER.includes(key)) return h >= 20;
  switch (key) {
    case "i": case "j": return !g.hasGap;
    case "b": case "d": case "k": return g.topSpan >= 0.4;
    case "h": return g.topSpan >= 0.33;
    case "l": return g.bottomSpan - g.topSpan > 0.15;
    case "f": return g.topInk - g.bottomInk >= 0.27;  // F: heavy full-width top bar
    case "g": return g.aspect >= 0.9 || g.midSpan < 0.65;  // G: wide, open in the middle
    case "p": return g.midSpan + g.wideRows < 1.44;    // p: bowl is a larger part of the letter
    case "q": return h >= 24.5;                        // Q (with tail) is taller than q
    case "t": return g.topInk >= 0.56;                 // T: heavy top bar
    case "y": return g.topInk + g.bottomInk < 1.15;    // Y: thin arms on top, thin stem below
    default: return null;
  }
}

function findGlyphs(rgba, width, height) {
  const size = width * height;
  const colors = new Int32Array(size);
  const counts = new Map();
  for (let i = 0, p = 0; i < size; i++, p += 4) {
    const color = (rgba[p] << 16) | (rgba[p + 1] << 8) | rgba[p + 2];
    colors[i] = color;
    counts.set(color, (counts.get(color) || 0) + 1);
  }
  const glyphs = [];
  const mask = new Uint8Array(size), core = new Uint8Array(size);
  for (const [color, count] of counts) {
    const r = color >> 16, g = (color >> 8) & 255, b = color & 255;
    if (count < 40 || Math.max(r, g, b) > 160) continue;   // characters are dark; noise is light
    for (let i = 0; i < size; i++) mask[i] = colors[i] === color ? 1 : 0;
    // 2x2 opening: keep pixels covered by a fully set 2x2 block (drops 1 px noise lines)
    core.fill(0);
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const i = y * width + x;
        if (mask[i] && mask[i + 1] && mask[i + width] && mask[i + width + 1]) {
          core[i] = core[i + 1] = core[i + width] = core[i + width + 1] = 1;
        }
      }
    }
    const glyph = measure(core, width, height);
    if (glyph) glyphs.push(glyph);
  }
  return glyphs.sort((a, b) => a.centerX - b.centerX);
}

function measure(core, width, height) {
  let x0 = width, x1 = -1, y0 = height, y1 = -1, n = 0, sumX = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!core[y * width + x]) continue;
      n++; sumX += x;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (n < 30) return null;
  const h = y1 - y0 + 1, w = x1 - x0 + 1, quarter = Math.max(Math.floor(h / 4), 1);
  const ink = [], span = [];
  for (let y = y0; y <= y1; y++) {
    let c = 0, left = -1, right = -1;
    for (let x = x0; x <= x1; x++) {
      if (!core[y * width + x]) continue;
      c++; if (left < 0) left = x; right = x;
    }
    ink.push(c); span.push(left < 0 ? 0 : right - left + 1);
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const maxInk = Math.max(...ink);
  return {
    centerX: sumX / n,
    hasGap: ink.includes(0),
    height: h,
    aspect: w / h,
    topInk: mean(ink.slice(0, quarter)) / maxInk,
    bottomInk: mean(ink.slice(-quarter)) / maxInk,
    topSpan: mean(span.slice(0, quarter)) / w,
    midSpan: mean(span.slice(quarter, h - quarter)) / w,
    bottomSpan: mean(span.slice(-quarter)) / w,
    wideRows: span.filter((v) => v > 0.6 * w).length / h,
  };
}

// ---- Pillow-exact preprocessing ------------------------------------------------------------------

function preprocess(rgba, width, height) {
  const outWidth = Math.trunc(width * (HEIGHT / height));  // ddddocr: int(w * 64 / h)
  const rgb = resizeLanczos(rgba, width, height, outWidth, HEIGHT);
  const data = new Float32Array(outWidth * HEIGHT);
  for (let i = 0, p = 0; i < data.length; i++, p += 3) {
    // Pillow RGB -> L: (R*19595 + G*38470 + B*7471 + 0x8000) >> 16
    data[i] = ((rgb[p] * 19595 + rgb[p + 1] * 38470 + rgb[p + 2] * 7471 + 0x8000) >>> 16) / 255;
  }
  return { data, width: outWidth };
}

const PRECISION_BITS = 22;                      // Pillow: 32 - 8 - 2
const ONE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);

function lanczos(x) {
  if (x < -3 || x >= 3) return 0;              // Pillow: -3 <= x < 3
  const sinc = (v) => (v === 0 ? 1 : Math.sin(Math.PI * v) / (Math.PI * v));
  return sinc(x) * sinc(x / 3);
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc (fixed-point weights per output pixel).
// Captchas are always the same size, so the weights are computed once and reused.
const coefficientCache = new Map();

function coefficients(inSize, outSize) {
  const cacheKey = inSize * 65536 + outSize;
  let cached = coefficientCache.get(cacheKey);
  if (!cached) coefficientCache.set(cacheKey, cached = computeCoefficients(inSize, outSize));
  return cached;
}

function computeCoefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 3 * filterScale;
  const result = [];
  for (let x = 0; x < outSize; x++) {
    const center = (x + 0.5) * scale;
    const min = Math.max(Math.trunc(center - support + 0.5), 0);
    const count = Math.min(Math.trunc(center + support + 0.5), inSize) - min;
    const weights = [];
    let total = 0;
    for (let i = 0; i < count; i++) {
      const w = lanczos((i + min - center + 0.5) / filterScale);
      weights.push(w);
      total += w;
    }
    result.push({
      min,
      weights: weights.map((w) => {
        const k = total !== 0 ? w / total : w;
        return k < 0 ? Math.trunc(k * ONE - 0.5) : Math.trunc(k * ONE + 0.5);
      }),
    });
  }
  return result;
}

const clip8 = (sum) => Math.min(255, Math.max(0, Math.floor(sum / ONE)));

// Horizontal pass, then vertical pass, 8-bit rounding in between - same order as Pillow.
function resizeLanczos(rgba, width, height, outWidth, outHeight) {
  const horizontal = coefficients(width, outWidth);
  const vertical = coefficients(height, outHeight);

  const temp = new Uint8Array(outWidth * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < outWidth; x++) {
      const { min, weights } = horizontal[x];
      let r = HALF, g = HALF, b = HALF;
      for (let i = 0; i < weights.length; i++) {
        const p = (y * width + min + i) * 4;
        r += rgba[p] * weights[i];
        g += rgba[p + 1] * weights[i];
        b += rgba[p + 2] * weights[i];
      }
      const q = (y * outWidth + x) * 3;
      temp[q] = clip8(r); temp[q + 1] = clip8(g); temp[q + 2] = clip8(b);
    }
  }

  const out = new Uint8Array(outWidth * outHeight * 3);
  for (let y = 0; y < outHeight; y++) {
    const { min, weights } = vertical[y];
    for (let x = 0; x < outWidth; x++) {
      let r = HALF, g = HALF, b = HALF;
      for (let i = 0; i < weights.length; i++) {
        const p = ((min + i) * outWidth + x) * 3;
        r += temp[p] * weights[i];
        g += temp[p + 1] * weights[i];
        b += temp[p + 2] * weights[i];
      }
      const q = (y * outWidth + x) * 3;
      out[q] = clip8(r); out[q + 1] = clip8(g); out[q + 2] = clip8(b);
    }
  }
  return out;
}
