#!/usr/bin/env python3
"""
SynthID Cleaner — drag-and-drop GUI around the V3 spectral bypass in this repo.

Removes Google SynthID's invisible watermark from images (e.g. AI-restored
old photographs), and optionally strips EXIF/XMP/IPTC metadata (including any
AI-generation provenance tags such as C2PA content credentials or IPTC
DigitalSourceType). No network calls, no telemetry — everything runs locally.
"""
import os
import sys
import threading
import traceback
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src" / "extraction"))

import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

from PIL import Image
from synthid_bypass import SynthIDBypass, SpectralCodebook
from robust_extractor import RobustSynthIDExtractor

CODEBOOK_PATH = REPO_DIR / "artifacts" / "spectral_codebook_v3.npz"
DETECTOR_CODEBOOK_PATH = REPO_DIR / "artifacts" / "codebook" / "robust_codebook.pkl"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def copy_pixels(src_path: str, dst_path: str):
    """Re-save the image unchanged (no spectral subtraction applied)."""
    im = Image.open(src_path)
    arr = np.array(im)
    Image.fromarray(arr, mode=im.mode).save(dst_path)


def strip_metadata(path: str):
    """Rebuild the file from raw pixel data so no EXIF/XMP/IPTC/ICC/GPS
    chunk survives — including any AI-generation provenance tags
    (e.g. C2PA content credentials, IPTC DigitalSourceType, Software tag).
    """
    im = Image.open(path)
    arr = np.array(im)
    clean = Image.fromarray(arr, mode=im.mode)
    clean.save(path)


def parse_dnd_paths(data: str):
    """Tk's dnd event data is a space-joined, brace-quoted path list."""
    paths, buf, in_brace = [], "", False
    for ch in data:
        if ch == "{":
            in_brace = True
        elif ch == "}":
            in_brace = False
        elif ch == " " and not in_brace:
            if buf:
                paths.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        paths.append(buf)
    return paths


class App:
    def __init__(self, root):
        self.root = root
        root.title("SynthID Cleaner")
        root.geometry("560x480")
        root.minsize(480, 400)

        self.strength = tk.StringVar(value="aggressive")
        self.strip_meta = tk.BooleanVar(value=True)
        self.out_dir = tk.StringVar(value=str(Path.home() / "Desktop" / "synthid-cleaned"))
        self.status = tk.StringVar(value="Loading codebook…")
        self.queue = []

        self._build_ui()
        self.root.after(100, self._load_codebook)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        drop_text = (
            "Drag images here\n(or click to choose files)"
            if HAS_DND else
            "Click to choose images"
        )
        self.drop = tk.Label(
            self.root, text=drop_text, relief="groove", bd=2,
            font=("Helvetica", 14), fg="#444", height=8, bg="#f4f4f4",
            cursor="hand2",
        )
        self.drop.pack(fill="both", expand=True, **pad)
        self.drop.bind("<Button-1>", lambda e: self._choose_files())

        if HAS_DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)

        row = tk.Frame(self.root)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Strength:").pack(side="left")
        ttk.OptionMenu(
            row, self.strength, self.strength.get(),
            "gentle", "moderate", "aggressive", "maximum",
        ).pack(side="left", padx=8)
        tk.Checkbutton(
            row, text="Strip EXIF/XMP/IPTC metadata",
            variable=self.strip_meta,
        ).pack(side="left", padx=8)

        out_row = tk.Frame(self.root)
        out_row.pack(fill="x", **pad)
        tk.Label(out_row, text="Save to:").pack(side="left")
        tk.Entry(out_row, textvariable=self.out_dir).pack(
            side="left", fill="x", expand=True, padx=8)
        tk.Button(out_row, text="Browse…", command=self._choose_out_dir).pack(side="left")

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", **pad)

        tk.Label(self.root, textvariable=self.status, anchor="w",
                 wraplength=520, justify="left").pack(fill="x", **pad)

        if not HAS_DND:
            tk.Label(
                self.root,
                text="(tkinterdnd2 not installed — using file picker instead of drag-and-drop)",
                fg="#888", font=("Helvetica", 10),
            ).pack(**pad)

    def _load_codebook(self):
        def work():
            try:
                self.bypass = SynthIDBypass()
                self.codebook = SpectralCodebook()
                self.codebook.load(str(CODEBOOK_PATH))
                self.detector = RobustSynthIDExtractor(
                    codebook_path=str(DETECTOR_CODEBOOK_PATH))
                self.root.after(0, lambda: self.status.set(
                    "Ready. Drag images in or click the box above."))
            except Exception as e:
                self.root.after(0, lambda: self.status.set(f"Failed to load codebook: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def _choose_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get() or str(Path.home()))
        if d:
            self.out_dir.set(d)

    def _choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")],
        )
        if paths:
            self._process(list(paths))

    def _on_drop(self, event):
        paths = [p for p in parse_dnd_paths(event.data)
                 if Path(p).suffix.lower() in IMAGE_EXTS]
        if paths:
            self._process(paths)

    def _process(self, paths):
        out_dir = Path(self.out_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        strength = self.strength.get()
        do_strip_meta = self.strip_meta.get()

        self.drop.config(state="disabled")
        self.progress.config(maximum=len(paths), value=0)

        def work():
            done = 0
            errors = []
            skipped = []
            for p in paths:
                src = Path(p)
                dst = out_dir / f"{src.stem}_clean{src.suffix}"
                self.root.after(0, lambda s=src.name: self.status.set(f"Checking {s}…"))
                try:
                    det = self.detector.detect(str(src))
                    if det.is_watermarked:
                        self.root.after(0, lambda s=src.name, c=det.confidence:
                                         self.status.set(f"Watermark found in {s} (conf {c:.2f}) — cleaning…"))
                        self.bypass.bypass_v3_file(
                            str(src), str(dst), self.codebook,
                            strength=strength, verify=False,
                        )
                        skipped.append(False)
                    else:
                        # No watermark detected: subtracting the codebook's carrier
                        # pattern anyway would imprint it onto a clean image instead
                        # of removing one, so just pass the image through untouched.
                        self.root.after(0, lambda s=src.name:
                                         self.status.set(f"No watermark in {s} — left untouched"))
                        copy_pixels(str(src), str(dst))
                        skipped.append(True)
                    if do_strip_meta:
                        strip_metadata(str(dst))
                except Exception as e:
                    errors.append(f"{src.name}: {e}")
                    traceback.print_exc()
                done += 1
                self.root.after(0, lambda d=done: self.progress.config(value=d))

            def finish():
                n_ok = done - len(errors)
                n_skipped = sum(skipped)
                msg = f"Done: {n_ok}/{len(paths)} processed → {out_dir}"
                if n_skipped:
                    msg += f"\n({n_skipped} had no watermark and were left untouched)"
                if errors:
                    msg += f"\n{len(errors)} failed: " + "; ".join(errors[:3])
                self.status.set(msg)
                self.drop.config(state="normal")
            self.root.after(0, finish)

        threading.Thread(target=work, daemon=True).start()


def main():
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
