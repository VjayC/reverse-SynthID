# SynthID Cleaner (GUI)

A drag and drop desktop app built on this repo's V3 spectral bypass
(`src/extraction/synthid_bypass.py`). Made for removing the SynthID
watermark left over after using generative AI to restore old photographs,
though it works on any Gemini generated image.

## Setup (one time only)

Double click `Launch SynthID Cleaner.command`. On first run it creates a
virtual environment and installs everything it needs automatically. This
takes a minute or two depending on your connection.

If you would rather do it by hand from the terminal:

```bash
cd gui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-gui.txt
```

## Using it

1. Double click `Launch SynthID Cleaner.command`. A small window opens.
2. Drag your images into the box, or click the box to pick files from a
   dialog instead.
3. Pick a strength from the dropdown: gentle, moderate, aggressive, or
   maximum. Aggressive is the default and works well for most photos.
4. Leave "Strip EXIF/XMP/IPTC metadata" checked if you also want camera and
   software tags, plus any AI provenance metadata, removed from the output
   files.
5. Set the output folder in the "Save to" field. It defaults to a
   synthid-cleaned folder on your Desktop.
6. Watch the status line at the bottom while it works. For each image it
   reports whether a watermark was found and removed, or whether the image
   was already clean and left untouched.

Each output file keeps the original name with `_clean` added, so
`photo.png` becomes `photo_clean.png` in the output folder.

## What it actually does

Every image is checked first with this repo's `RobustSynthIDExtractor`.
Only images where a watermark is actually detected go through the spectral
bypass. Images with no detectable watermark are copied through unchanged
instead, so the tool cannot accidentally stamp a watermark shaped pattern
onto a photo that never had one.

Everything runs locally: no network calls, no telemetry. It uses the V3
pipeline only (numpy, scipy, opencv, PyWavelets, scikit-learn), so no
PyTorch install or GPU is required.

## Notes

- Codebooks are read straight from this repo: `../artifacts/spectral_codebook_v3.npz`
  for the bypass, `../artifacts/codebook/robust_codebook.pkl` for detection.
  Nothing extra to download.
- Subject to this repo's [LICENSE](../LICENSE): non commercial use, with
  required attribution to the original author.
