#!/usr/bin/env python3
"""
=============================================================
  SMART DOOR ACCESS SYSTEM — ECAPA-TDNN Speaker Verification
  Computer Vision Course Project
  
  Upgraded from SVM → SpeechBrain ECAPA-TDNN (State of the Art)
  Pretrained on VoxCeleb2 (~1M utterances, 5994 speakers)

  HOW IT WORKS:
    - ECAPA-TDNN encodes any audio clip into a 192-dim speaker
      embedding (a "voice fingerprint")
    - Authorized speakers are enrolled from LibriSpeech files
      automatically on first run
    - Access is granted by cosine similarity between the test
      embedding and enrolled speaker embeddings
    - No SVM, no hand-crafted MFCC features — end-to-end neural

  AUTHORIZED speakers (DOOR OPENS):
    1272, 1673, 1988, 1993, 2035

  UNAUTHORIZED speakers (DOOR LOCKED):
    All other speakers in dev-clean
=============================================================
"""

import os
import sys
import glob
import random
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from scipy.fft import fft, fftfreq
import torch
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── SpeechBrain ECAPA-TDNN ─────────────────────────────────
try:
    from speechbrain.inference.speaker import SpeakerRecognition, EncoderClassifier
    SPEECHBRAIN_OK = True
except ImportError:
    try:
        from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
        SPEECHBRAIN_OK = True
    except ImportError:
        print("\n  [ERROR] SpeechBrain not installed.")
        print("  Run:  pip install speechbrain")
        sys.exit(1)


# ── CONFIGURATION ──────────────────────────────────────────
_BASE        = "/Users/stitli/Downloads/dev-clean/LibriSpeech"
DATASET_PATH = (os.path.join(_BASE, "dev-clean")
                if os.path.isdir(os.path.join(_BASE, "dev-clean")) else _BASE)

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR      = os.path.join(_SCRIPT_DIR, "outputs")
MODELS_DIR       = os.path.join(_SCRIPT_DIR, "models")
ENROLLED_PATH    = os.path.join(MODELS_DIR, "enrolled_speakers.pkl")
REG_USERS_PATH   = os.path.join(MODELS_DIR, "registered_users_ecapa.pkl")
ECAPA_CACHE_DIR  = os.path.join(MODELS_DIR, "speechbrain_cache")

# Authorized LibriSpeech speaker IDs
AUTHORIZED_SPEAKERS = {'1272', '1673', '1988', '1993', '2035'}

SAMPLE_RATE       = 16000
RECORD_SECONDS    = 5           # slightly longer for better embeddings
N_REG_SAMPLES     = 4           # clips per registration

# Cosine similarity thresholds
# ECAPA embeddings are much more discriminative than raw MFCCs
# so thresholds are lower (not 0.99x)
SIM_THRESHOLD_ENROLLED  = 0.75  # for mic-registered users
SIM_THRESHOLD_LIBRI     = 0.70  # for auto-enrolled LibriSpeech speakers

N_LIBRI_ENROLL_FILES    = 10    # LibriSpeech files used to build each speaker's mean embedding


# ── UTILITIES ──────────────────────────────────────────────

def open_file(path):
    try:
        subprocess.Popen(['open', path])
    except Exception:
        pass


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b  = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── ECAPA-TDNN MODEL ───────────────────────────────────────

def load_ecapa_model():
    """
    Load pretrained ECAPA-TDNN from SpeechBrain (VoxCeleb2).
    Downloads ~80 MB on first run, cached locally afterwards.
    """
    print("\n  Loading ECAPA-TDNN (pretrained on VoxCeleb2)...")
    os.makedirs(ECAPA_CACHE_DIR, exist_ok=True)

    try:
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=ECAPA_CACHE_DIR,
            run_opts={"device": "cpu"},
        )
        print("  ✔  ECAPA-TDNN loaded successfully.")
        return encoder
    except Exception as e:
        print(f"  [ERROR] Could not load ECAPA-TDNN: {e}")
        sys.exit(1)


def get_embedding(encoder, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Given a raw audio waveform (numpy float32), return a 192-dim
    ECAPA-TDNN speaker embedding (L2-normalised).
    """
    # SpeechBrain expects a torch tensor of shape [1, N_samples]
    wav_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        # encode_batch returns [batch, 1, embedding_dim]
        embedding = encoder.encode_batch(wav_tensor)
    emb = embedding.squeeze().cpu().numpy()
    # L2-normalise so cosine sim == dot product
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def get_embedding_from_file(encoder, filepath: str) -> np.ndarray:
    """Load a .flac/.wav file and return its ECAPA embedding."""
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    audio     = librosa.util.normalize(audio)
    audio, _  = librosa.effects.trim(audio, top_db=20)
    if len(audio) < SAMPLE_RATE * 0.5:
        raise ValueError("Audio too short")
    return get_embedding(encoder, audio)


# ── SPEAKER ENROLLMENT (LibriSpeech authorized speakers) ───

def enroll_authorized_speakers(encoder):
    """
    Auto-enroll the 5 authorized LibriSpeech speakers by averaging
    ECAPA embeddings from N_LIBRI_ENROLL_FILES clips each.
    Saves to ENROLLED_PATH.
    """
    print("\n" + "="*62)
    print("  AUTO-ENROLLING AUTHORIZED SPEAKERS (LibriSpeech)")
    print("="*62)

    if not os.path.isdir(DATASET_PATH):
        print(f"\n  Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    enrolled = {}
    for spk_id in sorted(AUTHORIZED_SPEAKERS):
        spk_path = os.path.join(DATASET_PATH, spk_id)
        if not os.path.isdir(spk_path):
            print(f"  [WARN] Speaker {spk_id} not found in dataset, skipping.")
            continue

        files = glob.glob(os.path.join(spk_path, '**', '*.flac'), recursive=True)
        files = files[:N_LIBRI_ENROLL_FILES]
        if not files:
            print(f"  [WARN] No .flac files for speaker {spk_id}, skipping.")
            continue

        embeddings = []
        ok = 0
        for f in files:
            try:
                emb = get_embedding_from_file(encoder, f)
                embeddings.append(emb)
                ok += 1
            except Exception:
                pass

        if not embeddings:
            print(f"  [WARN] Could not extract any embeddings for {spk_id}.")
            continue

        mean_emb = np.mean(embeddings, axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)  # re-normalise mean
        enrolled[spk_id] = {
            'embedding': mean_emb,
            'n_clips':   ok,
            'source':    'librispeech',
        }
        print(f"  ✔  Speaker {spk_id}  enrolled from {ok} clips  "
              f"(emb norm = {np.linalg.norm(mean_emb):.3f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(enrolled, ENROLLED_PATH)
    print(f"\n  Saved enrolled speakers → {ENROLLED_PATH}")
    return enrolled


def load_enrolled_speakers():
    if os.path.exists(ENROLLED_PATH):
        return joblib.load(ENROLLED_PATH)
    return {}


def load_registered_users():
    if os.path.exists(REG_USERS_PATH):
        return joblib.load(REG_USERS_PATH)
    return {}


def save_registered_users(users: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(users, REG_USERS_PATH)


# ── VISUALIZATIONS (same as original, kept intact) ─────────

def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y    = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.5:
        raise ValueError("Too short")
    return y


def save_4panel_analysis(audio, title, filename, sr=SAMPLE_RATE, auto_open=False):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f'Audio Analysis — {title}', fontsize=14, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Waveform
    ax1 = fig.add_subplot(gs[0, 0])
    t   = np.linspace(0, len(audio)/sr, len(audio))
    ax1.plot(t, audio, color='#2196F3', lw=0.6)
    ax1.fill_between(t, audio, alpha=0.15, color='#2196F3')
    ax1.set_title('Waveform', fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.25)

    # STFT Spectrogram
    ax2  = fig.add_subplot(gs[0, 1])
    D_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img  = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz',
                                     ax=ax2, cmap='magma')
    ax2.set_title('STFT Spectrogram', fontweight='bold')
    plt.colorbar(img, ax=ax2, format='%+2.0f dB', pad=0.02)

    # FFT
    ax3 = fig.add_subplot(gs[1, 0])
    n   = len(audio)
    yf  = np.abs(fft(audio))[:n//2]
    xf  = fftfreq(n, 1/sr)[:n//2]
    ax3.plot(xf, yf, color='#FF5722', lw=0.7)
    ax3.fill_between(xf, yf, alpha=0.25, color='#212121')
    ax3.set_title('FFT — Frequency Domain', fontweight='bold')
    ax3.set_xlabel('Frequency (Hz)'); ax3.set_ylabel('Magnitude')
    ax3.set_xlim([0, sr/2]); ax3.grid(True, alpha=0.25)

    # MFCC heatmap
    ax4  = fig.add_subplot(gs[1, 1])
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    img2 = librosa.display.specshow(mfcc, sr=sr, x_axis='time',
                                     ax=ax4, cmap='RdBu_r')
    ax4.set_title('MFCC Coefficients', fontweight='bold')
    ax4.set_ylabel('MFCC index')
    plt.colorbar(img2, ax=ax4, pad=0.02)

    out = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")
    if auto_open:
        open_file(out)


def save_mel_spectrogram(audio, title, filename, sr=SAMPLE_RATE, auto_open=False):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                       ax=ax, cmap='viridis', fmax=8000)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(f'Mel Spectrogram — {title}', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")
    if auto_open:
        open_file(out)


def save_embedding_similarity_chart(similarities: dict, decision: int,
                                     best_name: str, best_sim: float,
                                     auto_open: bool = True):
    """
    Bar chart showing cosine similarity of the test embedding
    against every enrolled speaker. Replaces the old SVM probability bar.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    names = list(similarities.keys())
    sims  = [similarities[n] for n in names]
    colors = []
    for n, s in zip(names, sims):
        if n == best_name and decision == 1:
            colors.append('#66BB6A')   # green = matched
        elif s >= SIM_THRESHOLD_ENROLLED or s >= SIM_THRESHOLD_LIBRI:
            colors.append('#FFA726')   # orange = above threshold but not chosen
        else:
            colors.append('#EF5350')   # red = no match

    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.4), 4))
    bars = ax.bar(names, sims, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

    # Threshold lines
    ax.axhline(SIM_THRESHOLD_LIBRI,    color='orange', ls='--', lw=1.2,
               label=f'LibriSpeech threshold ({SIM_THRESHOLD_LIBRI})')
    ax.axhline(SIM_THRESHOLD_ENROLLED, color='blue',   ls='--', lw=1.2,
               label=f'Registered threshold ({SIM_THRESHOLD_ENROLLED})')

    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Speaker')
    ax.set_ylabel('Cosine Similarity (ECAPA-TDNN embedding)')
    decision_str = 'DOOR OPEN 🟢' if decision == 1 else 'DOOR LOCKED 🔴'
    ax.set_title(f'ECAPA-TDNN Access Decision: {decision_str}  '
                 f'[best={best_name}, sim={best_sim:.4f}]',
                 fontweight='bold',
                 color='green' if decision == 1 else 'red')

    for bar, val in zip(bars, sims):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, 'last_access_decision.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")
    if auto_open:
        open_file(out)


def save_enrollment_overview(enrolled: dict, registered: dict):
    """Bar chart summarising all enrolled embeddings (norms as quality proxy)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    all_names  = []
    all_norms  = []
    all_colors = []

    for spk_id, data in enrolled.items():
        all_names.append(f"Spk {spk_id}\n(LibriSpeech)")
        all_norms.append(float(np.linalg.norm(data['embedding'])))
        all_colors.append('#42A5F5')

    for name, data in registered.items():
        all_names.append(f"{name}\n(Mic)")
        all_norms.append(float(np.linalg.norm(data['embedding'])))
        all_colors.append('#AB47BC')

    if not all_names:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(all_names)*1.4), 4))
    ax.bar(all_names, all_norms, color=all_colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Embedding L2 Norm')
    ax.set_title('Enrolled Speaker Embeddings (ECAPA-TDNN)', fontweight='bold')
    ax.axhline(1.0, color='red', ls='--', lw=1, label='Expected norm=1')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, 'enrolled_speakers.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")
    open_file(out)


# ── CORE CLASSIFICATION ────────────────────────────────────

def classify_audio(encoder, audio: np.ndarray, source_label: str = ""):
    """
    1. Extract ECAPA-TDNN embedding from test audio.
    2. Compare against ALL enrolled LibriSpeech authorized speakers.
    3. Compare against mic-registered custom users.
    4. Grant access if any similarity exceeds its threshold.
    """
    print(f"\n  Extracting ECAPA-TDNN embedding for: {source_label}")
    save_4panel_analysis(audio, f"Test — {source_label}",
                         "test_voice_analysis.png", auto_open=True)
    save_mel_spectrogram(audio, f"Test — {source_label}",
                         "test_voice_mel.png")

    test_emb = get_embedding(encoder, audio)
    print(f"  Embedding shape: {test_emb.shape}  |  norm = {np.linalg.norm(test_emb):.4f}")

    enrolled   = load_enrolled_speakers()
    registered = load_registered_users()

    similarities = {}
    best_name, best_sim, best_threshold = "none", -1, SIM_THRESHOLD_LIBRI

    print("\n  ── LibriSpeech authorized speakers ───────────────────")
    print(f"  {'Speaker':<22} {'Cosine Sim':>12}  {'Status'}")
    print(f"  {'-'*52}")
    for spk_id, data in enrolled.items():
        sim    = cosine_sim(test_emb, data['embedding'])
        status = "✅ MATCH" if sim >= SIM_THRESHOLD_LIBRI else "  ✗"
        print(f"  Spk {spk_id:<18} {sim:>12.5f}  {status}")
        similarities[f"Spk {spk_id}"] = sim
        if sim > best_sim:
            best_sim, best_name, best_threshold = sim, f"Spk {spk_id}", SIM_THRESHOLD_LIBRI

    if registered:
        print("\n  ── Mic-registered users ──────────────────────────────")
        print(f"  {'Name':<22} {'Cosine Sim':>12}  {'Status'}")
        print(f"  {'-'*52}")
        for name, data in registered.items():
            sim    = cosine_sim(test_emb, data['embedding'])
            status = "✅ MATCH" if sim >= SIM_THRESHOLD_ENROLLED else "  ✗"
            print(f"  {name:<22} {sim:>12.5f}  {status}")
            similarities[name] = sim
            if sim > best_sim:
                best_sim, best_name, best_threshold = sim, name, SIM_THRESHOLD_ENROLLED

    # Decision
    decision = 1 if best_sim >= best_threshold else 0

    print(f"\n  Best match : {best_name}  (sim = {best_sim:.5f}, "
          f"threshold = {best_threshold})")
    if decision == 1:
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║   ✅  ACCESS GRANTED — DOOR OPEN     ║")
        print(f"  ║   Welcome, {best_name:<25}║")
        print(f"  ╚══════════════════════════════════════╝")
    else:
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║   🔴  ACCESS DENIED  — DOOR LOCKED   ║")
        print(f"  ╚══════════════════════════════════════╝")

    save_embedding_similarity_chart(similarities, decision,
                                     best_name, best_sim, auto_open=True)
    return decision


# ── MICROPHONE INPUT ───────────────────────────────────────

def record_audio(seconds: int = RECORD_SECONDS) -> np.ndarray:
    """Record from mic, return normalised numpy float32 array."""
    rec   = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = librosa.util.normalize(rec.flatten())
    return audio


def test_with_mic(encoder):
    print("\n" + "="*62)
    print("  DOOR ACCESS TEST — Microphone")
    print("="*62)
    input(f"\n  Press Enter then speak clearly for {RECORD_SECONDS} seconds ...")
    audio = record_audio()
    print("  Recording complete.")
    return classify_audio(encoder, audio, source_label="Microphone")


# ── DATASET FILE TEST ──────────────────────────────────────

def test_with_dataset_file(encoder):
    print("\n" + "="*62)
    print("  DOOR ACCESS TEST — Dataset File")
    print("="*62)
    print("\n  Authorized IDs (expect DOOR OPEN)  : "
          + ", ".join(sorted(AUTHORIZED_SPEAKERS)))
    all_spk = sorted([d for d in os.listdir(DATASET_PATH)
                       if os.path.isdir(os.path.join(DATASET_PATH, d))])
    unauth  = [s for s in all_spk if s not in AUTHORIZED_SPEAKERS][:5]
    print("  Unauthorized IDs (expect LOCKED)   : " + ", ".join(unauth) + " ...")

    speaker_id   = input("\n  Enter speaker ID to test: ").strip()
    speaker_path = os.path.join(DATASET_PATH, speaker_id)
    if not os.path.isdir(speaker_path):
        print(f"  Speaker '{speaker_id}' not found."); return

    files = glob.glob(os.path.join(speaker_path, '**', '*.flac'), recursive=True)
    if not files:
        print("  No .flac files found."); return

    # Use a file NOT in the enrollment set for a fair test
    enrolled_files_used = files[:N_LIBRI_ENROLL_FILES]
    test_candidates = [f for f in files if f not in enrolled_files_used]
    chosen = random.choice(test_candidates if test_candidates else files)

    true_label = "AUTHORIZED" if speaker_id in AUTHORIZED_SPEAKERS else "UNAUTHORIZED"
    print(f"\n  File      : {os.path.basename(chosen)}")
    print(f"  Speaker   : {speaker_id}  ({true_label})")
    print(f"  Note      : Using a file NOT in the enrollment set for a fair test")

    audio    = load_audio(chosen)
    result   = classify_audio(encoder, audio,
                               source_label=f"Speaker {speaker_id} ({true_label})")
    expected = 1 if speaker_id in AUTHORIZED_SPEAKERS else 0
    if result == expected:
        print(f"\n  ✔  Correct prediction!")
    else:
        print(f"\n  ✗  Wrong prediction (expected {'OPEN' if expected==1 else 'LOCKED'}).")


# ── MIC REGISTRATION ───────────────────────────────────────

def register_new_user(encoder):
    """
    Record N_REG_SAMPLES clips, average ECAPA embeddings, save.
    The average embedding is L2-re-normalised before storing.
    """
    print("\n" + "="*62)
    print("  USER REGISTRATION — ECAPA-TDNN")
    print(f"  Recording {N_REG_SAMPLES} clips × {RECORD_SECONDS}s each.")
    print("  Tip: speak naturally, vary sentence content slightly.")
    print("="*62)

    name = input("\n  Your name: ").strip()
    if not name:
        print("  Name cannot be empty."); return

    all_embs   = []
    last_audio = None

    for i in range(N_REG_SAMPLES):
        input(f"\n  [{i+1}/{N_REG_SAMPLES}] Press Enter then speak for {RECORD_SECONDS}s ...")
        last_audio = record_audio()
        emb        = get_embedding(encoder, last_audio)
        all_embs.append(emb)
        print(f"  Clip {i+1} recorded  (emb norm = {np.linalg.norm(emb):.4f})")

    mean_emb = np.mean(all_embs, axis=0)
    mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)

    # Self-consistency check
    print(f"\n  Self-consistency check (cosine sim vs mean embedding):")
    for i, emb in enumerate(all_embs):
        sim = cosine_sim(emb, mean_emb)
        print(f"    Clip {i+1}: {sim:.5f}  "
              + ("✅ Good" if sim > 0.90 else "⚠  Low — consider re-recording"))

    # Save visuals
    safe_name = name.replace(' ', '_')
    save_4panel_analysis(last_audio, f"Registration — {name}",
                         f"register_{safe_name}.png", auto_open=True)
    save_mel_spectrogram(last_audio, f"Registration — {name}",
                         f"register_{safe_name}_mel.png")

    users = load_registered_users()
    users[name] = {
        'embedding': mean_emb,
        'n_clips':   N_REG_SAMPLES,
        'source':    'microphone',
    }
    save_registered_users(users)

    print(f"\n  ✔  '{name}' registered successfully.")
    print(f"  Access threshold : {SIM_THRESHOLD_ENROLLED}")
    print(f"  All registered   : {list(users.keys())}")


# ── MAIN MENU ──────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("   SMART DOOR ACCESS — ECAPA-TDNN Speaker Verification")
    print("   Computer Vision Course Project")
    print("="*62)
    print(f"\n  Model    : ECAPA-TDNN (pretrained on VoxCeleb2)")
    print(f"  Backend  : SpeechBrain")
    print(f"  Outputs  → {OUTPUTS_DIR}")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    # Load ECAPA-TDNN
    encoder = load_ecapa_model()

    # Auto-enroll authorized LibriSpeech speakers if not done yet
    enrolled = load_enrolled_speakers()
    if not enrolled:
        print("\n  No enrolled speakers found → running auto-enrollment...")
        enrolled = enroll_authorized_speakers(encoder)
    else:
        print(f"\n  Loaded {len(enrolled)} enrolled LibriSpeech speakers: "
              + ", ".join(sorted(enrolled.keys())))

    # Offer mic registration
    if input("\n  Register a new mic user? (y/n): ").strip().lower() == 'y':
        register_new_user(encoder)

    while True:
        print("\n" + "-"*52)
        print("   MENU")
        print("-"*52)
        print("  1 · Test with MICROPHONE (speak now)")
        print("  2 · Test with DATASET FILE (demo authorized/denied)")
        print("  3 · Register / re-register a mic user")
        print("  4 · Show enrolled & registered speakers")
        print("  5 · Re-enroll LibriSpeech authorized speakers")
        print("  6 · Show embedding overview chart")
        print("  7 · Open outputs folder in Finder")
        print("  8 · Exit")

        choice = input("\n  Choice (1-8): ").strip()

        if choice == '1':
            test_with_mic(encoder)

        elif choice == '2':
            test_with_dataset_file(encoder)

        elif choice == '3':
            register_new_user(encoder)

        elif choice == '4':
            enrolled   = load_enrolled_speakers()
            registered = load_registered_users()
            print(f"\n  LibriSpeech authorized ({len(enrolled)}):")
            for spk_id, data in enrolled.items():
                print(f"    · Spk {spk_id}  [{data['n_clips']} clips enrolled]")
            print(f"\n  Mic-registered users ({len(registered)}):")
            if registered:
                for name, data in registered.items():
                    print(f"    · {name}  [{data['n_clips']} clips]")
            else:
                print("    (none)")
            print(f"\n  Thresholds: LibriSpeech={SIM_THRESHOLD_LIBRI}  "
                  f"Mic={SIM_THRESHOLD_ENROLLED}")

        elif choice == '5':
            if input("  Re-enroll all authorized speakers? (y/n): ").strip().lower() == 'y':
                enrolled = enroll_authorized_speakers(encoder)

        elif choice == '6':
            enrolled   = load_enrolled_speakers()
            registered = load_registered_users()
            save_enrollment_overview(enrolled, registered)

        elif choice == '7':
            subprocess.Popen(['open', OUTPUTS_DIR])
            print(f"  Opening {OUTPUTS_DIR} in Finder...")

        elif choice == '8':
            print("\n  Goodbye!\n"); break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
