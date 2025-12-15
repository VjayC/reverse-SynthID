<p align="center">
  <img src="assets/synthid-watermark.jpeg" alt="SynthID Watermark Analysis" width="100%">
</p>

<h1 align="center">🔍 SynthID Watermark Reverse Engineering</h1>

<p align="center">
  <b>Discovering Google's hidden AI watermark patterns through signal analysis</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-Research-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=flat-square" alt="Accuracy">
</p>

---

## 🎯 Overview

This project reverse-engineers **Google's SynthID watermarking technology** by analyzing 250 AI-generated images from Gemini. Since the neural network encoder/decoder is proprietary, we use signal processing techniques to discover the watermark's structure.

### Key Discovery

SynthID uses **spread-spectrum phase encoding** in the frequency domain—not LSB replacement or simple noise addition. The watermark embeds information through precise phase relationships at specific carrier frequencies.

## 🔬 Discovered Patterns

| Carrier Frequency | Phase Coherence | Description |
|:----------------:|:---------------:|:------------|
| **(±14, ±14)** | 99.99% | Primary diagonal carrier |
| **(±126, ±14)** | 99.97% | Secondary horizontal |
| **(±98, ±14)** | 99.94% | Tertiary carrier |
| **(±128, ±128)** | 99.92% | Center frequency |
| **(±210, ±14)** | 99.77% | Extended carrier |
| **(±238, ±14)** | 99.71% | Edge carrier |

### Detection Metrics
- **Noise Correlation**: ~0.218 between watermarked images
- **Structure Ratio**: ~1.32
- **Detection Threshold**: correlation > 0.179

## 🖼️ Extracted Watermark Visualizations

<table>
<tr>
<td width="50%">

**Enhanced Visualization (500x Amplification)**
<img src="artifacts/visualizations/synthid_watermark_amp500x.png" width="100%">

</td>
<td width="50%">

**Frequency Domain Carriers**
<img src="artifacts/visualizations/synthid_watermark_frequency.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**False Color (HSV Encoding)**
<img src="artifacts/visualizations/synthid_watermark_falsecolor.png" width="100%">

</td>
<td width="50%">

**Phase Encoding Pattern**
<img src="artifacts/visualizations/synthid_watermark_phase.png" width="100%">

</td>
</tr>
</table>

## 📁 Project Structure

```
synthid-demarker/
├── 📄 README.md                    # This file
├── 📋 requirements.txt             # Python dependencies
│
├── 💻 src/
│   ├── analysis/
│   │   ├── synthid_codebook_finder.py    # Pattern discovery
│   │   └── deep_synthid_analysis.py      # Frequency analysis
│   └── extraction/
│       └── synthid_codebook_extractor.py # Codebook extraction & detection
│
├── 🎯 artifacts/
│   ├── codebook/
│   │   ├── synthid_codebook.pkl          # Extracted codebook (9 MB)
│   │   └── synthid_codebook_meta.json    # Carrier frequencies
│   └── visualizations/                   # Watermark images
│
├── 📂 data/
│   └── pure_white/                       # 250 Gemini AI images
│
├── 📚 docs/
│   └── SYNTHID_CODEBOOK_ANALYSIS.md      # Technical documentation
│
└── 🖼️ assets/
    └── synthid-watermark.jpeg            # Cover image
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/synthid-demarker.git
cd synthid-demarker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Detect Watermark

```bash
python src/extraction/synthid_codebook_extractor.py detect "path/to/image.png" \
    --codebook "artifacts/codebook/synthid_codebook.pkl"
```

**Output:**
```
Detection Results:
  Watermarked: True
  Confidence: 1.0000
  Correlation: 0.5355
  Phase Match: 0.9571
  Structure Ratio: 1.2753
```

### Extract New Codebook

```bash
python src/extraction/synthid_codebook_extractor.py extract "data/pure_white/" \
    --output "./my_codebook.pkl"
```

### Run Analysis

```bash
# Comprehensive pattern discovery
python src/analysis/synthid_codebook_finder.py

# Deep frequency analysis
python src/analysis/deep_synthid_analysis.py
```

## 🧠 How It Works

### 1. Pattern Discovery
Analyze noise patterns across multiple images to find consistent structures that persist despite varying image content.

### 2. Frequency Analysis
Use FFT to identify carrier frequencies where the watermark is embedded through phase modulation.

### 3. Phase Coherence
Measure phase consistency at carrier frequencies—high coherence indicates watermark presence.

### 4. Codebook Extraction
Build reference patterns from averaged signals across many watermarked images.

### 5. Detection
Compare test image against codebook using correlation, phase matching, and structure ratio metrics.

## 📊 Technical Details

### Watermark Characteristics
- **Embedding Domain**: Frequency (FFT phase)
- **Signal Strength**: ~0.1-0.15 pixel values
- **Carrier Count**: 100+ frequency locations
- **Robustness**: Survives moderate compression

### Detection Algorithm
```python
def detect_synthid(image, codebook):
    # 1. Extract noise pattern
    noise = image - denoise(image)
    
    # 2. Check carrier phase coherence
    fft = fft2(noise)
    phase_match = check_phases(fft, codebook.carriers)
    
    # 3. Correlate with reference
    correlation = correlate(noise, codebook.reference)
    
    # 4. Apply decision thresholds
    is_watermarked = (
        correlation > 0.179 and 
        phase_match > 0.5 and 
        0.8 < structure_ratio < 1.8
    )
    
    return is_watermarked, confidence
```

## 📚 References

- [SynthID: Identifying AI-generated images](https://deepmind.google/technologies/synthid/)
- [Nature Paper: Scalable watermarking for AI-generated images](https://doi.org/10.1038/s41586-024-07754-z)
- [Spread Spectrum Watermarking](https://en.wikipedia.org/wiki/Digital_watermarking)

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. SynthID is proprietary technology owned by Google DeepMind. The extracted patterns and detection methods are intended for:

- Academic research on watermarking techniques
- Security analysis of AI-generated content identification
- Understanding spread-spectrum encoding methods

## 📄 License

Research and educational use only. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with 🔬 by reverse engineering enthusiasts
</p>
