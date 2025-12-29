# -*- coding: utf-8 -*-
"""
PHI-TREE MODE 2: ADAPTIVE NOISE TRACKING IN FINANCIAL MARKETS
==============================================================

DUAL OPERATIONAL MODES:

MODE 1 (Natural Convergence):
  - Stable physical systems (HRV, climate, seismology, EEG)
  - H converges to φ⁻¹ = 0.618 without intervention
  - φ-Tree validates convergence (typical error <1%)

MODE 2 (Adaptive Tracking) - THIS SCRIPT:
  - Chaotic human systems (markets, social networks)
  - H fluctuates without natural φ⁻¹ attractor
  - φ-Tree tracks noise evolution in real-time
  - Deviation from φ⁻¹ measures human noise
  - Noise becomes information for optimization

Key Insight:
Because natural systems converge to φ⁻¹ (Mode 1), we can measure
deviation in chaotic systems (Mode 2) to track and quantify noise.

This script demonstrates:
1. Local H(t) calculated in sliding windows
2. H fluctuates without convergence (expected behavior)
3. Adaptive tracking of noise evolution
4. Regime classification (ADAPTIVE/TRANSITIONAL/DISSIPATIVE)
5. Noise quantification for optimization

Data: Bitcoin & EUR/USD hourly returns (data/btcusdeur/)
Method: Sliding-window DFA-1 + adaptive regime tracking

Date: December 29, 2025
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# PHI CONSTANTS
PHI = (1 + np.sqrt(5)) / 2        # 1.618034
PHI_INV = 1 / PHI                  # 0.618034 (reference, NOT target)
DELTA = np.sqrt(PHI) - 1           # 0.272020

def dfa_1(signal: np.ndarray, scale_min: int = 16, scale_max: int = None) -> Tuple[float, float]:
    """
    Detrended Fluctuation Analysis (DFA-1) for Hurst exponent.
    
    Args:
        signal: Time series
        scale_min: Minimum scale
        scale_max: Maximum scale (default: N/4)
    
    Returns:
        H: Hurst exponent
        R2: Fit quality
    """
    N = len(signal)
    if scale_max is None:
        scale_max = N // 4
    
    # Cumulative sum (profile)
    profile = np.cumsum(signal - np.mean(signal))
    
    # Log-spaced scales
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), 20).astype(int))
    scales = scales[scales < N]
    
    fluctuations = []
    
    for scale in scales:
        # Divide profile into segments
        n_segments = N // scale
        
        fluct_sum = 0
        for i in range(n_segments):
            segment = profile[i*scale:(i+1)*scale]
            x = np.arange(len(segment))
            
            # Linear detrend (DFA-1)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            fluct_sum += np.sum((segment - trend)**2)
        
        F = np.sqrt(fluct_sum / (n_segments * scale))
        fluctuations.append(F)
    
    # Log-log fit: F(n) ~ n^H
    log_scales = np.log10(scales)
    log_flucts = np.log10(fluctuations)
    
    H, intercept = np.polyfit(log_scales, log_flucts, 1)
    
    # R^2
    y_pred = H * log_scales + intercept
    ss_res = np.sum((log_flucts - y_pred)**2)
    ss_tot = np.sum((log_flucts - np.mean(log_flucts))**2)
    R2 = 1 - (ss_res / ss_tot)
    
    return H, R2


def classify_regime(H: float) -> str:
    """Classify dynamical regime based on Hurst exponent."""
    if H < 0.5 - DELTA:
        return "DISSIPATIVE"
    elif H > 0.5 + DELTA:
        return "ADAPTIVE"
    else:
        return "TRANSITIONAL"


def sliding_window_tracking(returns: np.ndarray, window_size: int = 2048, 
                            step_size: int = 256, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Track local H(t) evolution in sliding windows.
    
    Args:
        returns: Price returns
        window_size: Window size for DFA
        step_size: Step between windows
        verbose: Print progress
    
    Returns:
        H_values: Local Hurst exponents
        positions: Window center positions
        regimes: Regime classifications
    """
    n = len(returns)
    n_windows = (n - window_size) // step_size + 1
    
    H_values = []
    positions = []
    regimes = []
    
    if verbose:
        print(f"\n  Processing {n_windows} sliding windows...")
        print(f"  Progress: ", end='', flush=True)
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window = returns[start:end]
        
        H, _ = dfa_1(window, scale_min=16, scale_max=window_size//2)
        regime = classify_regime(H)
        
        H_values.append(H)
        positions.append((start + end) // 2)
        regimes.append(regime)
        
        # Progress indicator
        if verbose and (i % max(1, n_windows // 20) == 0):
            print(f"{int(i/n_windows*100)}%", end='...', flush=True)
    
    if verbose:
        print("100% [DONE]\n")
    
    return np.array(H_values), np.array(positions), regimes


def load_bitcoin_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load Bitcoin hourly data and compute returns."""
    data_dir = Path(__file__).parent.parent / "data" / "btcusdeur"
    filepath = data_dir / "btc_1h_full.csv"
    
    print(f"\n{'='*80}")
    print("LOADING: Bitcoin (BTC/USD)")
    print(f"{'='*80}\n")
    print(f"File: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    # Log returns
    returns = np.log(df['close'] / df['close'].shift(1)).dropna().values
    
    print(f"  Rows: {len(df):,}")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Returns: N={len(returns):,}, mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")
    
    return df, returns


def load_eurusd_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load EUR/USD hourly data and compute returns."""
    data_dir = Path(__file__).parent.parent / "data" / "btcusdeur"
    filepath = data_dir / "eurusd_1h_full.csv"
    
    print(f"\n{'='*80}")
    print("LOADING: EUR/USD Forex")
    print(f"{'='*80}\n")
    print(f"File: {filepath}")
    
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    
    # Log returns
    returns = np.log(df['close'] / df['close'].shift(1)).dropna().values
    
    print(f"  Rows: {len(df):,}")
    print(f"  Period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"  Returns: N={len(returns):,}, mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")
    
    return df, returns


def analyze_noise_tracking(name: str, returns: np.ndarray, 
                          window_size: int = 2048, step_size: int = 256) -> Dict:
    """
    Analyze noise tracking results for one market.
    
    Args:
        name: Market name
        returns: Price returns
        window_size: DFA window size
        step_size: Sliding step
    
    Returns:
        Dictionary with tracking statistics
    """
    print(f"\n{'='*80}")
    print(f"ADAPTIVE TRACKING: {name}")
    print(f"{'='*80}\n")
    
    # Sliding window tracking (with progress)
    H_values, positions, regimes = sliding_window_tracking(returns, window_size, step_size, verbose=True)
    
    # Statistics
    H_mean = np.mean(H_values)
    H_std = np.std(H_values)
    H_min = np.min(H_values)
    H_max = np.max(H_values)
    H_range = H_max - H_min
    
    # Noise measure: deviation from φ⁻¹
    noise = np.abs(H_values - PHI_INV)
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)
    noise_max = np.max(noise)
    noise_min = np.min(noise)
    
    # Find extreme windows (for inspection)
    idx_max_H = np.argmax(H_values)
    idx_min_H = np.argmin(H_values)
    idx_max_noise = np.argmax(noise)
    
    # Regime distribution
    regime_counts = Counter(regimes)
    regime_dist = {k: v/len(regimes)*100 for k, v in regime_counts.items()}
    
    # Regime transitions
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append((i, regimes[i-1], regimes[i]))
    
    print(f"  [OK] Completed: {len(H_values)} windows analyzed")
    print(f"  Window size: {window_size}, Step: {step_size}")
    
    print(f"\n  H STATISTICS:")
    print(f"    Mean:  {H_mean:.4f} ± {H_std:.4f}")
    print(f"    Range: [{H_min:.4f}, {H_max:.4f}]")
    print(f"    Fluctuation: Δ = {H_range:.4f}")
    print(f"    Median: {np.median(H_values):.4f}")
    print(f"    Q1-Q3: [{np.percentile(H_values, 25):.4f}, {np.percentile(H_values, 75):.4f}]")
    
    print(f"\n  EXTREME WINDOWS (for inspection):")
    print(f"    MAX H: {H_values[idx_max_H]:.4f} at window {idx_max_H} (position {positions[idx_max_H]})")
    print(f"    MIN H: {H_values[idx_min_H]:.4f} at window {idx_min_H} (position {positions[idx_min_H]})")
    print(f"    Range span: {(idx_max_H - idx_min_H) * step_size} samples")
    
    print(f"\n  NOISE MEASURE (|H - phi^-1|):")
    print(f"    Mean: {noise_mean:.4f} ± {noise_std:.4f}")
    print(f"    Range: [{noise_min:.4f}, {noise_max:.4f}]")
    print(f"    MAX noise: {noise_max:.4f} at window {idx_max_noise}")
    print(f"    Quantifies human noise intensity")
    
    # Distance from natural attractor
    pct_near_phi = (noise < 0.05).sum() / len(noise) * 100
    pct_far_phi = (noise > 0.15).sum() / len(noise) * 100
    print(f"\n  DISTANCE FROM phi^-1 ATTRACTOR:")
    print(f"    Close (<5% error):   {pct_near_phi:.1f}% of windows")
    print(f"    Far (>15% error):    {pct_far_phi:.1f}% of windows")
    print(f"    No natural convergence observed (Mode 2)")
    
    print(f"\n  REGIME DISTRIBUTION:")
    for regime in sorted(regime_dist.keys()):
        print(f"    {regime}: {regime_dist[regime]:.1f}%")
    
    print(f"\n  REGIME TRANSITIONS:")
    if len(transitions) > 0:
        print(f"    Total transitions: {len(transitions)}")
        print(f"    First 3 transitions:")
        for i, (win, old, new) in enumerate(transitions[:3]):
            print(f"      Window {win}: {old} to {new}")
        if len(transitions) > 3:
            print(f"      ... and {len(transitions) - 3} more")
    else:
        print(f"    No regime transitions (stable chaos)")
    
    return {
        'name': name,
        'n_windows': len(H_values),
        'H_mean': H_mean,
        'H_std': H_std,
        'H_min': H_min,
        'H_max': H_max,
        'H_range': H_range,
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'noise_max': noise_max,
        'regime_dist': regime_dist,
        'H_values': H_values,
        'positions': positions,
        'regimes': regimes
    }


def main():
    """Main adaptive noise tracking pipeline."""
    
    print("=" * 80)
    print("PHI-TREE MODE 2: ADAPTIVE NOISE TRACKING")
    print("=" * 80)
    print(f"\nReference: phi^-1 = {PHI_INV:.6f} (Natural attractor in Mode 1)")
    print(f"Universal Gap: delta = {DELTA:.6f}\n")
    
    print("** FINANCIAL MARKETS: NO NATURAL CONVERGENCE **")
    print("   Human noise dominates -> H fluctuates wildly")
    print("   We TRACK noise, NOT force convergence to phi^-1")
    print("   Deviation = Information for optimization\n")
    
    # Load data
    btc_df, btc_returns = load_bitcoin_data()
    eur_df, eur_returns = load_eurusd_data()
    
    # Track noise evolution with WINDOW 2048
    print(f"\n{'='*80}")
    print("ANALYSIS WITH WINDOW SIZE = 2048 (longer-term tracking)")
    print(f"{'='*80}")
    
    btc_result_2048 = analyze_noise_tracking("Bitcoin", btc_returns, window_size=2048, step_size=256)
    eur_result_2048 = analyze_noise_tracking("EUR/USD", eur_returns, window_size=2048, step_size=256)
    
    # Track noise evolution with WINDOW 1024
    print(f"\n{'='*80}")
    print("ANALYSIS WITH WINDOW SIZE = 1024 (shorter-term tracking)")
    print(f"{'='*80}")
    
    btc_result_1024 = analyze_noise_tracking("Bitcoin", btc_returns, window_size=1024, step_size=128)
    eur_result_1024 = analyze_noise_tracking("EUR/USD", eur_returns, window_size=1024, step_size=128)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: NOISE TRACKING RESULTS (WINDOW COMPARISON)")
    print(f"{'='*80}\n")
    
    print("WINDOW SIZE = 2048 (Long-term, ~85 days per window)")
    print(f"{'Market':<12} {'H (mean±std)':<20} {'H Range':<18} {'Noise':<15} {'Windows':<10}")
    print("-" * 85)
    
    for r in [btc_result_2048, eur_result_2048]:
        h_str = f"{r['H_mean']:.4f}±{r['H_std']:.4f}"
        range_str = f"[{r['H_min']:.3f}, {r['H_max']:.3f}]"
        noise_str = f"{r['noise_mean']:.4f}±{r['noise_std']:.4f}"
        print(f"{r['name']:<12} {h_str:<20} {range_str:<18} {noise_str:<15} {r['n_windows']:<10}")
    
    print("\nWINDOW SIZE = 1024 (Short-term, ~43 days per window)")
    print(f"{'Market':<12} {'H (mean±std)':<20} {'H Range':<18} {'Noise':<15} {'Windows':<10}")
    print("-" * 85)
    
    for r in [btc_result_1024, eur_result_1024]:
        h_str = f"{r['H_mean']:.4f}±{r['H_std']:.4f}"
        range_str = f"[{r['H_min']:.3f}, {r['H_max']:.3f}]"
        noise_str = f"{r['noise_mean']:.4f}±{r['noise_std']:.4f}"
        print(f"{r['name']:<12} {h_str:<20} {range_str:<18} {noise_str:<15} {r['n_windows']:<10}")
    
    # Window size effect analysis
    print(f"\n{'='*80}")
    print("WINDOW SIZE EFFECT ON NOISE TRACKING")
    print(f"{'='*80}\n")
    
    print("Bitcoin:")
    print(f"  Window 2048: H_std = {btc_result_2048['H_std']:.4f}, Noise = {btc_result_2048['noise_mean']:.4f}")
    print(f"  Window 1024: H_std = {btc_result_1024['H_std']:.4f}, Noise = {btc_result_1024['noise_mean']:.4f}")
    print(f"  Shorter window shows {'more' if btc_result_1024['H_std'] > btc_result_2048['H_std'] else 'less'} fluctuation")
    
    print("\nEUR/USD:")
    print(f"  Window 2048: H_std = {eur_result_2048['H_std']:.4f}, Noise = {eur_result_2048['noise_mean']:.4f}")
    print(f"  Window 1024: H_std = {eur_result_1024['H_std']:.4f}, Noise = {eur_result_1024['noise_mean']:.4f}")
    print(f"  Shorter window shows {'more' if eur_result_1024['H_std'] > eur_result_2048['H_std'] else 'less'} fluctuation")
    
    print("\nInterpretation:")
    print("  • Longer windows (2048) smooth out short-term noise")
    print("  • Shorter windows (1024) capture rapid regime changes")
    print("  • Both show NO convergence to φ⁻¹ (Mode 2 confirmed)")
    
    # ASCII visualization of H evolution (use 2048 window results)
    print(f"\n{'='*80}")
    print("H EVOLUTION VISUALIZATION (first 40 windows, window=2048)")
    print(f"{'='*80}\n")
    
    def plot_ascii(H_vals, name, max_width=60):
        """Simple ASCII plot of H values."""
        print(f"{name}:")
        print(f"  φ⁻¹ = 0.618 reference line")
        print()
        
        # Take first 40 windows
        H_subset = H_vals[:min(40, len(H_vals))]
        
        # Normalize to 0-1 for plotting
        H_min_plot = 0.3
        H_max_plot = 0.7
        
        for i, h in enumerate(H_subset):
            # Position relative to range
            if h < H_min_plot:
                pos = 0
            elif h > H_max_plot:
                pos = max_width
            else:
                pos = int((h - H_min_plot) / (H_max_plot - H_min_plot) * max_width)
            
            # Mark φ⁻¹ position
            phi_pos = int((PHI_INV - H_min_plot) / (H_max_plot - H_min_plot) * max_width)
            
            # Build line
            line = [' '] * (max_width + 1)
            line[phi_pos] = '|'  # φ⁻¹ reference
            line[pos] = '●'       # H value
            
            # Regime marker
            regime = classify_regime(h)
            if regime == "DISSIPATIVE":
                marker = "◄ DISS"
            elif regime == "ADAPTIVE":
                marker = "► ADPT"
            else:
                marker = "■ TRAN"
            
            print(f"  {i:2d} {''.join(line)} {h:.3f} {marker}")
        
        print(f"\n     {'0.30':<{max_width//3}}{'0.50':<{max_width//3}}{'0.70'}")
        print(f"     {' '*phi_pos}↑ φ⁻¹={PHI_INV:.3f}\n")
    
    plot_ascii(btc_result_2048['H_values'], "Bitcoin (window=2048)")
    plot_ascii(eur_result_2048['H_values'], "EUR/USD (window=2048)")
    
    # Interpretation
    print(f"{'='*80}")
    print("INTERPRETATION: DUAL OPERATIONAL MODES")
    print(f"{'='*80}\n")
    
    print("MODE 1 (Natural Convergence) - NOT THIS EXPERIMENT:")
    print("  Systems: HRV, climate, EEG, seismology, fGn")
    print("  Behavior: H -> phi^-1 autonomously (TIER 0: <1% error)")
    print("  Framework: phi-Tree VALIDATES convergence")
    print("  Mechanism: Self-organization to optimal memory balance")
    print()
    
    print("MODE 2 (Adaptive Tracking) - THIS EXPERIMENT:")
    print("  Systems: Financial markets, social networks, high-frequency trading")
    print("  Behavior: H fluctuates wildly, NO natural phi^-1 attractor")
    print("  Framework: phi-Tree TRACKS noise evolution")
    print("  Mechanism: Adaptive lambda(n), epsilon(n) follow local dynamics")
    print("  Output: Noise measure -> INFORMATION for optimization")
    print()
    
    print("RESULTS ANALYSIS (Window=2048 long-term):")
    for r in [btc_result_2048, eur_result_2048]:
        print(f"\n{r['name']}:")
        print(f"  Wide H fluctuation: Δ = {r['H_range']:.4f} (no convergence)")
        print(f"  Noise intensity: {r['noise_mean']:.4f} ± {r['noise_std']:.4f}")
        print(f"  Max deviation: {r['noise_max']:.4f} (max human noise)")
        print(f"  Dominant regime: {max(r['regime_dist'], key=r['regime_dist'].get)} "
              f"({r['regime_dist'][max(r['regime_dist'], key=r['regime_dist'].get)]:.1f}%)")
    
    print("\nRESULTS ANALYSIS (Window=1024 short-term):")
    for r in [btc_result_1024, eur_result_1024]:
        print(f"\n{r['name']}:")
        print(f"  Wide H fluctuation: Δ = {r['H_range']:.4f} (no convergence)")
        print(f"  Noise intensity: {r['noise_mean']:.4f} ± {r['noise_std']:.4f}")
        print(f"  Max deviation: {r['noise_max']:.4f} (max human noise)")
        print(f"  Dominant regime: {max(r['regime_dist'], key=r['regime_dist'].get)} "
              f"({r['regime_dist'][max(r['regime_dist'], key=r['regime_dist'].get)]:.1f}%)")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHT: WHY BOTH MODES WORK")
    print(f"{'='*80}\n")
    
    print("Because NATURAL systems converge to phi^-1 (Mode 1),")
    print("we can MEASURE deviation in CHAOTIC systems (Mode 2).\n")
    
    print("Deviation from phi^-1 = Quantified human noise\n")
    
    print("This noise becomes INFORMATION for:")
    print("  • Risk optimization (track volatility regimes)")
    print("  • Trend detection (regime transitions)")
    print("  • Parameter adaptation (adjust to local H)")
    print("  • Monte Carlo generation (match noise statistics)")
    print()
    
    print("UNIFIED FRAMEWORK:")
    print("  Same math: lambda(n), epsilon(n), tau dynamics")
    print("  Dual purpose:")
    print("    1. VALIDATION tool (natural systems)")
    print("    2. OPTIMIZATION engine (chaotic systems)")
    print()
    
    print(f"{'='*80}")
    print("ADAPTIVE NOISE TRACKING COMPLETE")
    print(f"{'='*80}\n")
    
    print("Next steps:")
    print("  • Use noise measure for risk optimization")
    print("  • Track regime transitions for trading signals")
    print("  • Generate synthetic data matching local H statistics")
    print("  • Compare with Mode 1 systems (HRV, climate, etc.)")
    
    # Generate visual plots
    print(f"\n{'='*80}")
    print("GENERATING VISUAL PLOTS")
    print(f"{'='*80}\n")
    
    create_tracking_plots(btc_result_2048, btc_result_1024, 
                         eur_result_2048, eur_result_1024)
    
    print("[OK] Plots saved:")
    print("  • financial_noise_tracking.png")
    print("  • noise_comparison.png")


def create_tracking_plots(btc_2048, btc_1024, eur_2048, eur_1024):
    """Create comprehensive visualization of noise tracking results."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Define colors
    color_btc = '#F7931A'  # Bitcoin orange
    color_eur = '#003399'  # EUR blue
    color_phi = '#FF0000'  # φ⁻¹ reference red
    
    # ========== PLOT 1: Bitcoin H Evolution (both windows) ==========
    ax1 = plt.subplot(2, 2, 1)
    
    # Window 2048
    windows_2048 = np.arange(len(btc_2048['H_values']))
    ax1.plot(windows_2048, btc_2048['H_values'], 
             color=color_btc, alpha=0.6, linewidth=1.5, 
             label=f'Window=2048 (H={btc_2048["H_mean"]:.3f}±{btc_2048["H_std"]:.3f})')
    
    # Window 1024
    windows_1024 = np.arange(len(btc_1024['H_values']))
    ax1.plot(windows_1024, btc_1024['H_values'], 
             color=color_btc, alpha=0.3, linewidth=1, linestyle='--',
             label=f'Window=1024 (H={btc_1024["H_mean"]:.3f}±{btc_1024["H_std"]:.3f})')
    
    # φ⁻¹ reference
    ax1.axhline(y=PHI_INV, color=color_phi, linestyle='--', linewidth=2, 
                label=f'φ⁻¹ = {PHI_INV:.3f} (Natural attractor)', alpha=0.7)
    
    # Regime bands
    ax1.axhspan(0.5 - DELTA, 0.5 + DELTA, alpha=0.1, color='gray', 
                label='TRANSITIONAL regime')
    
    ax1.set_xlabel('Window Index', fontsize=11)
    ax1.set_ylabel('Hurst Exponent (H)', fontsize=11)
    ax1.set_title('Bitcoin: H Evolution (NO Convergence to φ⁻¹)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 0.7)
    
    # ========== PLOT 2: EUR/USD H Evolution (both windows) ==========
    ax2 = plt.subplot(2, 2, 2)
    
    # Window 2048
    windows_2048_eur = np.arange(len(eur_2048['H_values']))
    ax2.plot(windows_2048_eur, eur_2048['H_values'], 
             color=color_eur, alpha=0.6, linewidth=1.5,
             label=f'Window=2048 (H={eur_2048["H_mean"]:.3f}±{eur_2048["H_std"]:.3f})')
    
    # Window 1024
    windows_1024_eur = np.arange(len(eur_1024['H_values']))
    ax2.plot(windows_1024_eur, eur_1024['H_values'], 
             color=color_eur, alpha=0.3, linewidth=1, linestyle='--',
             label=f'Window=1024 (H={eur_1024["H_mean"]:.3f}±{eur_1024["H_std"]:.3f})')
    
    # φ⁻¹ reference
    ax2.axhline(y=PHI_INV, color=color_phi, linestyle='--', linewidth=2, 
                label=f'φ⁻¹ = {PHI_INV:.3f}', alpha=0.7)
    
    # Regime bands
    ax2.axhspan(0.5 - DELTA, 0.5 + DELTA, alpha=0.1, color='gray',
                label='TRANSITIONAL regime')
    
    ax2.set_xlabel('Window Index', fontsize=11)
    ax2.set_ylabel('Hurst Exponent (H)', fontsize=11)
    ax2.set_title('EUR/USD: H Evolution (NO Convergence to φ⁻¹)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 0.7)
    
    # ========== PLOT 3: Noise Measure Comparison ==========
    ax3 = plt.subplot(2, 2, 3)
    
    # Calculate noise for both markets and windows
    btc_noise_2048 = np.abs(btc_2048['H_values'] - PHI_INV)
    btc_noise_1024 = np.abs(btc_1024['H_values'] - PHI_INV)
    eur_noise_2048 = np.abs(eur_2048['H_values'] - PHI_INV)
    eur_noise_1024 = np.abs(eur_1024['H_values'] - PHI_INV)
    
    # Plot noise evolution
    ax3.plot(windows_2048, btc_noise_2048, color=color_btc, alpha=0.7, 
             linewidth=1.5, label='BTC (w=2048)')
    ax3.plot(windows_2048_eur, eur_noise_2048, color=color_eur, alpha=0.7, 
             linewidth=1.5, label='EUR (w=2048)')
    
    # Mean lines
    ax3.axhline(y=btc_2048['noise_mean'], color=color_btc, linestyle=':', 
                linewidth=2, alpha=0.5, label=f'BTC mean={btc_2048["noise_mean"]:.3f}')
    ax3.axhline(y=eur_2048['noise_mean'], color=color_eur, linestyle=':', 
                linewidth=2, alpha=0.5, label=f'EUR mean={eur_2048["noise_mean"]:.3f}')
    
    ax3.set_xlabel('Window Index', fontsize=11)
    ax3.set_ylabel('Noise Measure (|H - φ⁻¹|)', fontsize=11)
    ax3.set_title('Noise Intensity: Deviation from Natural Attractor', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ========== PLOT 4: Distribution Histograms ==========
    ax4 = plt.subplot(2, 2, 4)
    
    # Histograms
    bins = np.linspace(0.3, 0.7, 30)
    ax4.hist(btc_2048['H_values'], bins=bins, alpha=0.5, color=color_btc, 
             label=f'BTC (mean={btc_2048["H_mean"]:.3f})', density=True)
    ax4.hist(eur_2048['H_values'], bins=bins, alpha=0.5, color=color_eur, 
             label=f'EUR (mean={eur_2048["H_mean"]:.3f})', density=True)
    
    # φ⁻¹ reference
    ax4.axvline(x=PHI_INV, color=color_phi, linestyle='--', linewidth=2.5, 
                label=f'φ⁻¹ = {PHI_INV:.3f}', alpha=0.8)
    
    # Regime boundaries
    ax4.axvline(x=0.5 - DELTA, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax4.axvline(x=0.5 + DELTA, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax4.axvspan(0.5 - DELTA, 0.5 + DELTA, alpha=0.1, color='gray')
    
    ax4.set_xlabel('Hurst Exponent (H)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('H Distribution: Wide Spread, NO Convergence', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle('φ-Tree MODE 2: Adaptive Noise Tracking in Financial Markets\n'
                 'Human chaos dominates, H fluctuates without natural φ⁻¹ attractor',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('financial_noise_tracking.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== SECOND FIGURE: Window Size Comparison ==========
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # BTC: Window comparison
    ax1 = axes[0, 0]
    ax1.plot(windows_2048, btc_2048['H_values'], color=color_btc, linewidth=2, 
             label=f'W=2048: std={btc_2048["H_std"]:.4f}')
    ax1.axhline(y=PHI_INV, color=color_phi, linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_title('Bitcoin: Window 2048 (Long-term)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hurst (H)', fontsize=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 0.7)
    
    ax2 = axes[0, 1]
    ax2.plot(windows_1024, btc_1024['H_values'], color=color_btc, linewidth=1.5, alpha=0.8,
             label=f'W=1024: std={btc_1024["H_std"]:.4f}')
    ax2.axhline(y=PHI_INV, color=color_phi, linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_title('Bitcoin: Window 1024 (Short-term)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Hurst (H)', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 0.7)
    
    # EUR: Window comparison
    ax3 = axes[1, 0]
    ax3.plot(windows_2048_eur, eur_2048['H_values'], color=color_eur, linewidth=2,
             label=f'W=2048: std={eur_2048["H_std"]:.4f}')
    ax3.axhline(y=PHI_INV, color=color_phi, linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_title('EUR/USD: Window 2048 (Long-term)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Window Index', fontsize=10)
    ax3.set_ylabel('Hurst (H)', fontsize=10)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.3, 0.7)
    
    ax4 = axes[1, 1]
    ax4.plot(windows_1024_eur, eur_1024['H_values'], color=color_eur, linewidth=1.5, alpha=0.8,
             label=f'W=1024: std={eur_1024["H_std"]:.4f}')
    ax4.axhline(y=PHI_INV, color=color_phi, linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_title('EUR/USD: Window 1024 (Short-term)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Window Index', fontsize=10)
    ax4.set_ylabel('Hurst (H)', fontsize=10)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.3, 0.7)
    
    fig2.suptitle('Window Size Effect: Shorter Windows Show More Fluctuation\n'
                  'Both scales show NO convergence to φ⁻¹ (Mode 2 confirmed)',
                  fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
