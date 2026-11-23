"""Build the release zips.

    python tools/build.py

dist/srmist-auto-captcha-chrome-vX.zip   Chrome / Edge (manifest.json as is)
dist/srmist-auto-captcha-firefox-vX.zip  Firefox (manifest with a background page instead of a service worker)

Firefox needs a different manifest: it has no service workers or offscreen API for extensions,
and Chrome reports Firefox's keys as errors, so each browser gets its own zip.
"""
import json
import os
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = ["background.js", "content.js", "offscreen.html", "offscreen.js", "solver.js",
         "solver-service.js", "icons", "lib", "model"]


def firefox_manifest(manifest):
    manifest = dict(manifest)
    manifest["background"] = {"scripts": ["background.js"], "type": "module"}
    manifest["permissions"] = [p for p in manifest["permissions"] if p != "offscreen"]
    # Firefox MV3 doesn't grant the content-script site automatically; ask for it at install time.
    manifest["host_permissions"] = [m.replace("srmiststudentportal/*", "*")
                                    for cs in manifest["content_scripts"] for m in cs["matches"]]
    manifest["browser_specific_settings"] = {"gecko": {
        "id": "srmist-auto-captcha@local",
        "strict_min_version": "140.0",
        "data_collection_permissions": {"required": ["none"]},
    }}
    return manifest


def write_zip(path, manifest, skip=()):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        for name in FILES:
            source = os.path.join(ROOT, name)
            paths = [source] if os.path.isfile(source) else [
                os.path.join(d, f) for d, _, fs in os.walk(source) for f in fs]
            for p in paths:
                arcname = os.path.relpath(p, ROOT).replace(os.sep, "/")
                if arcname not in skip:
                    z.write(p, arcname)
    print(f"{os.path.relpath(path, ROOT)}  {os.path.getsize(path) / 1e6:.1f} MB")


with open(os.path.join(ROOT, "manifest.json"), encoding="utf-8") as f:
    chrome = json.load(f)
dist = os.path.join(ROOT, "dist")
os.makedirs(dist, exist_ok=True)
version = chrome["version"]
write_zip(os.path.join(dist, f"srmist-auto-captcha-chrome-v{version}.zip"), chrome)
write_zip(os.path.join(dist, f"srmist-auto-captcha-firefox-v{version}.zip"), firefox_manifest(chrome),
          skip={"offscreen.html", "offscreen.js"})
