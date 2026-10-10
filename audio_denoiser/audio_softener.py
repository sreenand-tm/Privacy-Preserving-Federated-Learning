"""
audio_softener.py
================================================================================
Acoustic Artifact Diagnosis & Raised-Cosine Softening Engine
================================================================================
- Implements Raised-Cosine (Hann) Windowing & Overlap-Add (OLA) smoothing
- Applies High-Frequency Spectral Tapering (Raised-Cosine Roll-Off) to soften harsh edges
- Provides Temporal Gain Envelope Smoothing (prevents sudden gating clicks)
- Exports 3-panel comparison graph: Raw, Cleaned Without Softener, Cleaned With Softener
================================================================================
"""

import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_rms(audio):
    """Compute RMS energy in dBFS."""
    rms = np.sqrt(np.mean(np.square(audio))) + 1e-12
    return 20.0 * np.log10(rms)

def raised_cosine_spectral_filter(audio, sr, cutoff_low=4500.0, cutoff_high=8000.0, max_attenuation_db=15.0):
    """
    Applies a frequency-domain Raised-Cosine roll-off filter to soften sharp high-frequency
    metallic chirps, sibilance, and Bark-band resonant peaks without muffling lower speech harmonics.
    """
    n_samples = len(audio)
    fft_vals = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sr)
    
    response = np.ones_like(freqs, dtype=np.float32)
    min_gain = 10.0 ** (-max_attenuation_db / 20.0)
    
    transition_mask = (freqs >= cutoff_low) & (freqs <= cutoff_high)
    f_norm = (freqs[transition_mask] - cutoff_low) / (cutoff_high - cutoff_low)
    rc_curve = 0.5 * (1.0 + np.cos(np.pi * f_norm))
    response[transition_mask] = min_gain + (1.0 - min_gain) * rc_curve
    response[freqs > cutoff_high] = min_gain
    
    filtered_fft = fft_vals * response
    filtered_audio = np.fft.irfft(filtered_fft, n=n_samples)
    return filtered_audio.astype(np.float32)

def raised_cosine_overlap_add_smooth(audio, sr, frame_len_ms=10.0, overlap_pct=0.5):
    """
    Eliminates 100 Hz inter-frame boundary step discontinuities caused by rectangular frame stitching.
    Splits audio into overlapping frames, applies a Raised-Cosine (Hann) window, and performs Overlap-Add (OLA).
    """
    frame_size = int(sr * (frame_len_ms / 1000.0))
    if frame_size % 2 != 0:
        frame_size += 1
        
    hop_size = int(frame_size * (1.0 - overlap_pct))
    window = signal.windows.hann(frame_size, sym=False).astype(np.float32)
    
    pad_len = frame_size + (hop_size - (len(audio) % hop_size)) % hop_size
    padded_audio = np.pad(audio, (0, pad_len), mode='reflect')
    
    num_frames = (len(padded_audio) - frame_size) // hop_size + 1
    output = np.zeros(len(padded_audio), dtype=np.float32)
    window_sum = np.zeros(len(padded_audio), dtype=np.float32)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = padded_audio[start:end] * window
        output[start:end] += frame
        window_sum[start:end] += window
        
    nonzero_mask = window_sum > 1e-6
    output[nonzero_mask] /= window_sum[nonzero_mask]
    
    return output[:len(audio)]

def temporal_envelope_smoothing(audio, sr, attack_ms=5.0, release_ms=40.0):
    """
    Smooths sudden abrupt energy gating. Prevents 'swishing' and musical noise.
    """
    alpha_attack = np.exp(-1.0 / (sr * (attack_ms / 1000.0)))
    alpha_release = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
    
    rectified = np.abs(audio)
    smooth_env = np.zeros_like(rectified)
    
    curr = 0.0
    for i in range(len(rectified)):
        val = rectified[i]
        if val > curr:
            curr = alpha_attack * curr + (1.0 - alpha_attack) * val
        else:
            curr = alpha_release * curr + (1.0 - alpha_release) * val
        smooth_env[i] = curr
        
    envelope_gain = np.ones_like(audio)
    high_spike_mask = rectified > (smooth_env * 2.5 + 1e-4)
    envelope_gain[high_spike_mask] = (smooth_env[high_spike_mask] * 2.5) / rectified[high_spike_mask]
    
    return (audio * envelope_gain).astype(np.float32)

def soften_audio_pipeline(clean_audio, raw_audio, sr, context_class="general"):
    """
    Complete Multi-Stage Audio Softening Engine:
    1. Raised-Cosine Overlap-Add to eliminate frame boundary clicks (10ms / 100Hz transitions)
    2. Temporal Envelope Smoothing to soften abrupt gating jumps
    3. Raised-Cosine Spectral Softening to tame sharp metallic high-frequency harmonics (>4.5 kHz)
    4. Adaptive blend based on detected noise context
    """
    s1 = raised_cosine_overlap_add_smooth(clean_audio, sr, frame_len_ms=10.0, overlap_pct=0.5)
    s2 = temporal_envelope_smoothing(s1, sr, attack_ms=8.0, release_ms=45.0)
    
    ctx = context_class.lower()
    if "dog" in ctx or "siren" in ctx or "horn" in ctx:
        cutoff_low = 4200.0
        atten_db = 14.0
    elif "drill" in ctx or "jackhammer" in ctx:
        cutoff_low = 3800.0
        atten_db = 18.0
    elif "music" in ctx:
        cutoff_low = 5200.0
        atten_db = 10.0
    else:
        cutoff_low = 4500.0
        atten_db = 12.0
        
    softened = raised_cosine_spectral_filter(
        s2, sr, cutoff_low=cutoff_low, cutoff_high=8500.0, max_attenuation_db=atten_db
    )
    
    final_output = softened * 0.98 + raw_audio[:len(softened)] * 0.02
    
    max_val = np.max(np.abs(final_output))
    if max_val > 0.98:
        final_output = final_output * (0.95 / max_val)
        
    return final_output.astype(np.float32)

def generate_comparison_graph(raw_audio, clean_no_soft, clean_soft, sr, output_png_path, title_info="Audio Comparison"):
    """
    Generates a 3-panel comparison graph exclusively for:
    1. Raw Audio (Waveform + Spectrogram)
    2. Cleaned without Softener (Waveform + Spectrogram, showing sharp noise)
    3. Cleaned with Softener (Waveform + Spectrogram, showing softened result)
    """
    n_samples = min(len(raw_audio), len(clean_no_soft), len(clean_soft))
    raw = raw_audio[:n_samples]
    c_no_soft = clean_no_soft[:n_samples]
    c_soft = clean_soft[:n_samples]
    
    time_axis = np.linspace(0, n_samples / sr, n_samples)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [1, 2.4]})
    fig.patch.set_facecolor('#0f172a')
    
    plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.90, bottom=0.08, left=0.07, right=0.96)
    fig.suptitle(f"Audio Spectrum & Waveform Comparison\n{title_info}", 
                 fontsize=14, fontweight='bold', color='#38bdf8', y=0.97)
    
    configs = [
        ("1. Raw Audio (Original)", raw, '#ef4444', 0),
        ("2. Cleaned without Softener (Sharp High-Freq Noise Present)", c_no_soft, '#f59e0b', 1),
        ("3. Cleaned with Softener (Raised-Cosine Filter Applied)", c_soft, '#10b981', 2)
    ]
    
    for label, sig, color, row_idx in configs:
        ax_wave = axes[row_idx, 0]
        ax_spec = axes[row_idx, 1]
        
        ax_wave.set_facecolor('#1e293b')
        ax_spec.set_facecolor('#1e293b')
        
        # 1. Waveform Plot
        ax_wave.plot(time_axis, sig, color=color, linewidth=0.7, alpha=0.9)
        ax_wave.set_title(label, fontsize=10, fontweight='bold', color=color, pad=4)
        ax_wave.set_xlim(0, time_axis[-1])
        ax_wave.set_ylim(-1.0, 1.0)
        ax_wave.set_ylabel("Amplitude", fontsize=8.5, color='#cbd5e1')
        ax_wave.tick_params(colors='#94a3b8', labelsize=8)
        ax_wave.grid(True, linestyle='--', alpha=0.25, color='#64748b')
        if row_idx == 2:
            ax_wave.set_xlabel("Time (seconds)", fontsize=8.5, color='#cbd5e1')
            
        # 2. Spectrogram Plot
        n_fft = 1024
        hop = 256
        Pxx, freqs, bins, im = ax_spec.specgram(
            sig, NFFT=n_fft, Fs=sr, noverlap=n_fft-hop, cmap='magma', vmin=-80, vmax=0
        )
        ax_spec.set_title(f"Spectrogram: {label}", fontsize=10, fontweight='bold', color='#e2e8f0', pad=4)
        ax_spec.set_xlim(0, time_axis[-1])
        ax_spec.set_ylim(0, min(12000, sr // 2))
        ax_spec.set_ylabel("Freq (Hz)", fontsize=8.5, color='#cbd5e1')
        ax_spec.tick_params(colors='#94a3b8', labelsize=8)
        if row_idx == 2:
            ax_spec.set_xlabel("Time (seconds)", fontsize=8.5, color='#cbd5e1')
            
        if row_idx == 1:
            ax_spec.axhspan(4500, 8500, color='#38bdf8', alpha=0.18, linestyle=':')
            ax_spec.text(0.02, 7000, "<- Sharp Noise Region (>4.5 kHz)", color='#38bdf8', fontsize=8.5, 
                         fontweight='bold', bbox=dict(facecolor='#0f172a', edgecolor='#38bdf8', boxstyle='round,pad=0.2', alpha=0.85))
        elif row_idx == 2:
            ax_spec.text(0.02, 7000, "[OK] Softened via Raised-Cosine Filter", color='#10b981', fontsize=8.5, 
                         fontweight='bold', bbox=dict(facecolor='#0f172a', edgecolor='#10b981', boxstyle='round,pad=0.2', alpha=0.85))

    plt.savefig(output_png_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return output_png_path

def diagnose_sharp_noise_metrics(raw_audio, clean_std, clean_soft, sr):
    """
    Computes objective acoustic metrics:
    1. High-Frequency Band Power Ratio (>4.5 kHz vs total power)
    2. Peak Harshness (Crest Factor)
    3. Noise Attenuation & RMS levels
    """
    sos_hf = signal.butter(4, min(4500.0, sr*0.4), btype='highpass', fs=sr, output='sos')
    hf_std = signal.sosfilt(sos_hf, clean_std)
    hf_soft = signal.sosfilt(sos_hf, clean_soft)
    
    rms_std = np.sqrt(np.mean(clean_std**2)) + 1e-12
    rms_soft = np.sqrt(np.mean(clean_soft**2)) + 1e-12
    rms_hf_std = np.sqrt(np.mean(hf_std**2)) + 1e-12
    rms_hf_soft = np.sqrt(np.mean(hf_soft**2)) + 1e-12
    
    hf_power_ratio_std = (rms_hf_std / rms_std) * 100.0
    hf_power_ratio_soft = (rms_hf_soft / rms_soft) * 100.0
    
    crest_hf_std = (np.max(np.abs(hf_std)) / rms_hf_std) if rms_hf_std > 1e-6 else 0
    crest_hf_soft = (np.max(np.abs(hf_soft)) / rms_hf_soft) if rms_hf_soft > 1e-6 else 0
    
    hf_suppression_db = 20.0 * np.log10(rms_hf_std / rms_hf_soft)
    
    return {
        "hf_power_pct_standard": round(float(hf_power_ratio_std), 2),
        "hf_power_pct_softened": round(float(hf_power_ratio_soft), 2),
        "crest_factor_standard": round(float(crest_hf_std), 2),
        "crest_factor_softened": round(float(crest_hf_soft), 2),
        "hf_harshness_reduction_db": round(float(hf_suppression_db), 2),
        "rms_standard_dbfs": round(float(20*np.log10(rms_std)), 2),
        "rms_softened_dbfs": round(float(20*np.log10(rms_soft)), 2)
    }
