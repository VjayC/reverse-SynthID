"""
SynthID Watermark Extraction Benchmark Suite

Comprehensive benchmarking for watermark extraction and removal:
1. Detection accuracy across image types
2. Removal quality (PSNR, SSIM)
3. Re-detection test (verify watermark is removed)
4. Performance metrics

Usage:
    python benchmark_extraction.py --input-dir /path/to/images --codebook codebook.pkl
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_extractor import RobustSynthIDExtractor, DetectionResult
from watermark_remover import WatermarkRemover, RemovalResult


@dataclass
class BenchmarkResults:
    """Results from benchmarking run."""
    n_images: int
    detection_rate: float
    avg_confidence: float
    avg_correlation: float
    avg_phase_match: float
    
    removal_success_rate: float
    avg_psnr: float
    avg_ssim: float
    avg_confidence_drop: float
    re_detection_rate: float
    
    total_time: float
    avg_time_per_image: float
    
    details: Dict


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for SynthID extraction and removal.
    """
    
    def __init__(
        self,
        codebook_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize benchmark suite.
        
        Args:
            codebook_path: Path to pre-extracted codebook
            verbose: Print progress during benchmarking
        """
        self.verbose = verbose
        self.extractor = RobustSynthIDExtractor()
        self.remover = None
        
        if codebook_path and os.path.exists(codebook_path):
            self.extractor.load_codebook(codebook_path)
            self.remover = WatermarkRemover(extractor=self.extractor)
    
    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def load_images(
        self,
        image_dir: str,
        sample_size: Optional[int] = None,
        extensions: set = {'.png', '.jpg', '.jpeg', '.webp'}
    ) -> List[Tuple[str, np.ndarray]]:
        """Load images from directory."""
        images = []
        
        for fname in sorted(os.listdir(image_dir)):
            if os.path.splitext(fname)[1].lower() in extensions:
                path = os.path.join(image_dir, fname)
                img = cv2.imread(path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append((path, img_rgb))
                    
                    if sample_size and len(images) >= sample_size:
                        break
        
        return images
    
    def benchmark_detection(
        self,
        images: List[Tuple[str, np.ndarray]]
    ) -> Dict:
        """
        Benchmark detection accuracy.
        
        Returns:
            Dict with detection metrics
        """
        self.log(f"\n{'='*60}")
        self.log("DETECTION BENCHMARK")
        self.log(f"{'='*60}")
        
        results = []
        start_time = time.time()
        
        for i, (path, img) in enumerate(images):
            try:
                result = self.extractor.detect_array(img)
                results.append({
                    'path': path,
                    'is_watermarked': result.is_watermarked,
                    'confidence': result.confidence,
                    'correlation': result.correlation,
                    'phase_match': result.phase_match,
                    'structure_ratio': result.structure_ratio,
                    'carrier_strength': result.carrier_strength,
                })
            except Exception as e:
                self.log(f"  Error processing {path}: {e}")
                results.append({
                    'path': path,
                    'error': str(e)
                })
            
            if (i + 1) % 10 == 0:
                self.log(f"  Processed {i+1}/{len(images)} images...")
        
        elapsed = time.time() - start_time
        
        # Compute statistics
        valid_results = [r for r in results if 'error' not in r]
        detected = [r for r in valid_results if r['is_watermarked']]
        
        detection_rate = len(detected) / len(valid_results) if valid_results else 0
        avg_confidence = np.mean([r['confidence'] for r in valid_results]) if valid_results else 0
        avg_correlation = np.mean([r['correlation'] for r in valid_results]) if valid_results else 0
        avg_phase_match = np.mean([r['phase_match'] for r in valid_results]) if valid_results else 0
        
        self.log(f"\n  Detection Rate: {detection_rate:.1%}")
        self.log(f"  Avg Confidence: {avg_confidence:.4f}")
        self.log(f"  Avg Correlation: {avg_correlation:.4f}")
        self.log(f"  Avg Phase Match: {avg_phase_match:.4f}")
        self.log(f"  Time: {elapsed:.2f}s ({elapsed/len(images):.3f}s per image)")
        
        return {
            'n_images': len(images),
            'n_valid': len(valid_results),
            'n_detected': len(detected),
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'avg_correlation': avg_correlation,
            'avg_phase_match': avg_phase_match,
            'elapsed_seconds': elapsed,
            'results': results
        }
    
    def benchmark_removal(
        self,
        images: List[Tuple[str, np.ndarray]],
        output_dir: Optional[str] = None,
        save_samples: int = 5
    ) -> Dict:
        """
        Benchmark removal quality.
        
        Args:
            images: List of (path, image) tuples
            output_dir: Directory to save sample cleaned images
            save_samples: Number of sample images to save
            
        Returns:
            Dict with removal metrics
        """
        if self.remover is None:
            return {'error': 'No remover initialized (need codebook)'}
        
        self.log(f"\n{'='*60}")
        self.log("REMOVAL BENCHMARK")
        self.log(f"{'='*60}")
        
        results = []
        start_time = time.time()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, (path, img) in enumerate(images):
            try:
                result = self.remover.remove(img, verify=True)
                
                entry = {
                    'path': path,
                    'success': result.success,
                    'psnr': result.psnr,
                    'ssim': result.ssim,
                    'removal_confidence': result.removal_confidence,
                    'original_watermarked': result.original_detection.is_watermarked,
                    'original_confidence': result.original_detection.confidence,
                    'cleaned_watermarked': result.cleaned_detection.is_watermarked if result.cleaned_detection else None,
                    'cleaned_confidence': result.cleaned_detection.confidence if result.cleaned_detection else None,
                }
                results.append(entry)
                
                # Save sample outputs
                if output_dir and i < save_samples:
                    fname = os.path.basename(path)
                    out_path = os.path.join(output_dir, f"cleaned_{fname}")
                    cv2.imwrite(out_path, cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                self.log(f"  Error processing {path}: {e}")
                results.append({
                    'path': path,
                    'error': str(e)
                })
            
            if (i + 1) % 10 == 0:
                self.log(f"  Processed {i+1}/{len(images)} images...")
        
        elapsed = time.time() - start_time
        
        # Compute statistics
        valid_results = [r for r in results if 'error' not in r]
        successful = [r for r in valid_results if r['success']]
        re_detected = [r for r in valid_results if r.get('cleaned_watermarked', True)]
        
        removal_success_rate = len(successful) / len(valid_results) if valid_results else 0
        avg_psnr = np.mean([r['psnr'] for r in valid_results]) if valid_results else 0
        avg_ssim = np.mean([r['ssim'] for r in valid_results]) if valid_results else 0
        
        # Confidence drop
        conf_drops = []
        for r in valid_results:
            if r.get('cleaned_confidence') is not None:
                drop = r['original_confidence'] - r['cleaned_confidence']
                conf_drops.append(drop)
        avg_conf_drop = np.mean(conf_drops) if conf_drops else 0
        
        re_detection_rate = len(re_detected) / len(valid_results) if valid_results else 0
        
        self.log(f"\n  Removal Success Rate: {removal_success_rate:.1%}")
        self.log(f"  Avg PSNR: {avg_psnr:.2f} dB")
        self.log(f"  Avg SSIM: {avg_ssim:.4f}")
        self.log(f"  Avg Confidence Drop: {avg_conf_drop:.4f}")
        self.log(f"  Re-detection Rate: {re_detection_rate:.1%}")
        self.log(f"  Time: {elapsed:.2f}s ({elapsed/len(images):.3f}s per image)")
        
        return {
            'n_images': len(images),
            'n_valid': len(valid_results),
            'n_successful': len(successful),
            'removal_success_rate': removal_success_rate,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_confidence_drop': avg_conf_drop,
            're_detection_rate': re_detection_rate,
            'elapsed_seconds': elapsed,
            'results': results
        }
    
    def run_full_benchmark(
        self,
        image_dir: str,
        sample_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        save_report: Optional[str] = None
    ) -> BenchmarkResults:
        """
        Run complete benchmark suite.
        
        Args:
            image_dir: Directory containing watermarked images
            sample_size: Max images to test (None for all)
            output_dir: Directory to save cleaned samples
            save_report: Path to save JSON report
            
        Returns:
            BenchmarkResults
        """
        self.log(f"\n{'='*60}")
        self.log("SYNTHID EXTRACTION BENCHMARK SUITE")
        self.log(f"{'='*60}")
        self.log(f"Image directory: {image_dir}")
        self.log(f"Sample size: {sample_size or 'all'}")
        
        # Load images
        self.log("\nLoading images...")
        images = self.load_images(image_dir, sample_size)
        self.log(f"Loaded {len(images)} images")
        
        if not images:
            raise ValueError("No images found in directory")
        
        # Run benchmarks
        total_start = time.time()
        
        detection_results = self.benchmark_detection(images)
        removal_results = self.benchmark_removal(images, output_dir)
        
        total_time = time.time() - total_start
        
        # Compile results
        results = BenchmarkResults(
            n_images=len(images),
            detection_rate=detection_results['detection_rate'],
            avg_confidence=detection_results['avg_confidence'],
            avg_correlation=detection_results['avg_correlation'],
            avg_phase_match=detection_results['avg_phase_match'],
            
            removal_success_rate=removal_results.get('removal_success_rate', 0),
            avg_psnr=removal_results.get('avg_psnr', 0),
            avg_ssim=removal_results.get('avg_ssim', 0),
            avg_confidence_drop=removal_results.get('avg_confidence_drop', 0),
            re_detection_rate=removal_results.get('re_detection_rate', 0),
            
            total_time=total_time,
            avg_time_per_image=total_time / len(images),
            
            details={
                'detection': detection_results,
                'removal': removal_results
            }
        )
        
        # Print summary
        self.log(f"\n{'='*60}")
        self.log("BENCHMARK SUMMARY")
        self.log(f"{'='*60}")
        self.log(f"Images Tested: {results.n_images}")
        self.log(f"")
        self.log(f"Detection:")
        self.log(f"  Rate: {results.detection_rate:.1%}")
        self.log(f"  Confidence: {results.avg_confidence:.4f}")
        self.log(f"")
        self.log(f"Removal:")
        self.log(f"  Success Rate: {results.removal_success_rate:.1%}")
        self.log(f"  PSNR: {results.avg_psnr:.2f} dB")
        self.log(f"  SSIM: {results.avg_ssim:.4f}")
        self.log(f"  Re-detection: {results.re_detection_rate:.1%}")
        self.log(f"")
        self.log(f"Performance:")
        self.log(f"  Total Time: {results.total_time:.2f}s")
        self.log(f"  Per Image: {results.avg_time_per_image:.3f}s")
        self.log(f"{'='*60}")
        
        # Save report
        if save_report:
            report = asdict(results)
            # Remove large result arrays for JSON
            if 'details' in report:
                for key in ['detection', 'removal']:
                    if key in report['details']:
                        report['details'][key].pop('results', None)
            
            with open(save_report, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"\nReport saved to: {save_report}")
        
        return results


def compare_with_original(
    image_dir: str,
    original_codebook: str,
    robust_codebook: str,
    sample_size: int = 50
):
    """
    Compare original vs robust extractor performance.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Original vs Robust Extractor")
    print("=" * 60)
    
    # Original extractor (using same interface)
    from synthid_codebook_extractor import detect_synthid
    
    # Robust extractor
    robust = RobustSynthIDExtractor()
    robust.load_codebook(robust_codebook)
    
    # Load images
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    images = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in extensions:
            path = os.path.join(image_dir, fname)
            images.append(path)
            if len(images) >= sample_size:
                break
    
    print(f"Testing on {len(images)} images...")
    
    # Compare
    original_detected = 0
    robust_detected = 0
    
    for path in images:
        # Original
        try:
            orig_result = detect_synthid(path, original_codebook)
            if orig_result['is_watermarked']:
                original_detected += 1
        except:
            pass
        
        # Robust
        try:
            robust_result = robust.detect(path)
            if robust_result.is_watermarked:
                robust_detected += 1
        except:
            pass
    
    print(f"\nResults:")
    print(f"  Original Extractor: {original_detected}/{len(images)} ({100*original_detected/len(images):.1f}%)")
    print(f"  Robust Extractor: {robust_detected}/{len(images)} ({100*robust_detected/len(images):.1f}%)")
    print(f"  Improvement: {robust_detected - original_detected} more detected")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SynthID Extraction Benchmark Suite')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory with watermarked images')
    parser.add_argument('--codebook', type=str, required=True,
                       help='Path to codebook file')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of images to test (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save cleaned samples')
    parser.add_argument('--output-report', type=str, default='benchmark_results.json',
                       help='Path to save JSON report')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    suite = BenchmarkSuite(
        codebook_path=args.codebook,
        verbose=not args.quiet
    )
    
    results = suite.run_full_benchmark(
        image_dir=args.input_dir,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        save_report=args.output_report
    )
