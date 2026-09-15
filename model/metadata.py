"""
Bonus Module D - Provenance & Metadata evidence.

Reads signals that live NEXT TO the pixels and combines them with the
visual verdict:

  Camera-origin evidence (pushes toward "real"):
    - EXIF Make / Model (camera or phone manufacturer)
    - DateTimeOriginal, ExposureTime, FNumber, ISOSpeedRatings, FocalLength
    - GPS block
  AI-origin evidence (pushes toward "AI-generated"):
    - Stable Diffusion / ComfyUI / InvokeAI "parameters" or "prompt"
      PNG text chunks
    - Software / tool tags naming a known generator
    - C2PA / Content Credentials manifests (JUMBF box in JPEG, c2pa chunks),
      especially the "trainedAlgorithmicMedia" digital source type
    - XMP digitalsourcetype markers

IMPORTANT HONESTY NOTES (surfaced to the user in the app/CLI):
  - Metadata is evidence, not proof. It is trivially stripped (screenshots,
    re-saves, messaging apps) and can be forged. Absence of camera EXIF is
    therefore only WEAK evidence of anything.
  - Camera EXIF on an AI image can be faked; an AI-tool tag on a real photo
    is very unlikely. The log-odds weights below reflect that asymmetry.
"""
import json
import os
import re

AI_SOFTWARE_PATTERNS = [
    r"stable\s*diffusion", r"sdxl", r"comfyui", r"invokeai", r"novelai",
    r"midjourney", r"dall[\s\-·]?e", r"firefly", r"imagen", r"flux\.?1?",
    r"leonardo\.?ai", r"dreamstudio", r"automatic1111", r"a1111",
    r"stability\s*ai", r"gan", r"diffusion",
]
AI_CHUNK_KEYS = {"parameters", "prompt", "workflow", "negative_prompt",
                 "sd-metadata", "dream", "generation_data", "invokeai_metadata"}

CAMERA_EXIF_TAGS = {
    271: "Make", 272: "Model", 36867: "DateTimeOriginal",
    33434: "ExposureTime", 33437: "FNumber", 34855: "ISOSpeedRatings",
    37386: "FocalLength", 42036: "LensModel",
}
GPS_TAG = 34853


def _scan_ai_text(text: str):
    hits = []
    low = text.lower()
    for pat in AI_SOFTWARE_PATTERNS:
        m = re.search(pat, low)
        if m:
            hits.append(m.group(0))
    return hits


def analyze_metadata(image_path: str) -> dict:
    """Returns a dict:
        camera_evidence: list[str]   (EXIF fields that indicate a camera)
        ai_evidence:     list[str]   (tags/chunks that indicate a generator)
        c2pa:            'ai' | 'present' | None
        log_odds_shift:  float       (added to the visual log-odds)
        summary:         one-line human-readable summary
    Never raises: on any failure returns neutral evidence.
    """
    camera, ai, c2pa = [], [], None
    try:
        from PIL import Image, ExifTags  # noqa: F401
        with Image.open(image_path) as im:
            # ---- EXIF ----
            try:
                exif = im.getexif()
            except Exception:
                exif = None
            if exif:
                for tag_id, name in CAMERA_EXIF_TAGS.items():
                    val = exif.get(tag_id)
                    if val:
                        sval = str(val).strip()
                        if sval:
                            camera.append(f"{name}={sval[:40]}")
                            ai += [f"EXIF:{h}" for h in _scan_ai_text(sval)]
                if exif.get(GPS_TAG):
                    camera.append("GPS data present")
                software = exif.get(305)
                if software:
                    ai += [f"Software:{h}" for h in _scan_ai_text(str(software))]

            # ---- PNG text chunks (SD & friends write generation params) ----
            info = getattr(im, "info", {}) or {}
            for k, v in info.items():
                lk = str(k).lower()
                if lk in AI_CHUNK_KEYS:
                    ai.append(f"png-chunk:{lk}")
                elif isinstance(v, str) and len(v) < 20000:
                    hits = _scan_ai_text(v)
                    if hits:
                        ai.append(f"png-chunk:{lk}~{hits[0]}")
            # XMP
            xmp = info.get("XML:com.adobe.xmp") or info.get("xmp")
            if isinstance(xmp, (str, bytes)):
                x = xmp.decode("utf-8", "ignore") if isinstance(xmp, bytes) else xmp
                if "trainedalgorithmicmedia" in x.lower():
                    ai.append("XMP:digitalSourceType=trainedAlgorithmicMedia")
                    c2pa = "ai"
    except Exception:
        pass

    # ---- Raw scan for C2PA / JUMBF containers ----
    try:
        with open(image_path, "rb") as f:
            head = f.read(4 * 1024 * 1024)
        if b"c2pa" in head or b"jumb" in head.lower():
            if b"trainedAlgorithmicMedia" in head:
                c2pa = "ai"
                ai.append("C2PA:trainedAlgorithmicMedia")
            elif c2pa is None:
                c2pa = "present"
    except Exception:
        pass

    # ---- Evidence -> log-odds shift (asymmetric on purpose) ----
    shift = 0.0
    if ai:
        shift += 4.0          # a generator tag is near-conclusive
    if camera and not ai:
        # need real camera fields, not just any EXIF byte
        strong = [c for c in camera if c.split("=")[0] in
                  ("Make", "Model", "DateTimeOriginal", "ExposureTime",
                   "FNumber", "ISOSpeedRatings", "LensModel") or "GPS" in c]
        if len(strong) >= 2:
            shift -= 2.2      # strong but forgeable -> smaller magnitude
        elif len(strong) == 1:
            shift -= 1.0

    if ai:
        summary = "Metadata contains AI-generator traces (" + ", ".join(sorted(set(ai))[:3]) + ")."
    elif camera:
        summary = "Camera EXIF present (" + ", ".join(camera[:3]) + ") - consistent with a real photo, though EXIF can be forged."
    else:
        summary = "No metadata evidence (common for screenshots and re-saved images); verdict relies on visual analysis."

    return {
        "camera_evidence": camera,
        "ai_evidence": sorted(set(ai)),
        "c2pa": c2pa,
        "log_odds_shift": shift,
        "summary": summary,
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(analyze_metadata(sys.argv[1]), indent=2))
