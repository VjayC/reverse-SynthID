# SynthID Cleaner (GUI)

A small drag-and-drop desktop app around this repo's V3 spectral bypass
(`src/extraction/synthid_bypass.py`). Useful for e.g. removing the SynthID
watermark left behind after using generative AI to restore old photographs.

- Drop images in, get cleaned copies out — no command line needed after setup.
- Runs `RobustSynthIDExtractor` on each image first. The spectral bypass only
  runs if a watermark is actually detected; otherwise the image is passed
  through untouched. (Blindly subtracting the codebook's carrier pattern from
  an image that never had it would imprint a watermark-shaped artifact
  instead of removing one — this is why detection gates the bypass.)
- Optionally strips all EXIF/XMP/IPTC metadata from the output (including any
  AI-generation provenance tags such as C2PA content credentials or IPTC
  `DigitalSourceType`), by rebuilding the file from raw pixel data.
- Runs fully offline. No network calls, no telemetry.
- Uses V3 (pure signal processing — numpy/scipy/opencv/PyWavelets/
  scikit-learn only, no PyTorch required, no GPU needed).

## Setup

```bash
cd gui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-gui.txt
```

## Run

Double-click `Launch SynthID Cleaner.command`, or from the terminal:

```bash
cd gui
source venv/bin/activate
python gui.py
```

The first double-click sets up the venv automatically if it doesn't exist yet.

## Notes

- Reads the bypass codebook from `../artifacts/spectral_codebook_v3.npz` and
  the detector codebook from `../artifacts/codebook/robust_codebook.pkl`
  (both already in this repo) — no extra downloads.
- Subject to this repo's [LICENSE](../LICENSE): non-commercial use, with
  required attribution to the original author.
