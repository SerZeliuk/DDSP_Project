# audio_processor.py
import numpy as np
import librosa
import crepe

class AudioProcessor:
    """
    Load audio and extract frame-level f0 (via CREPE) and loudness (via RMS → dB).
    """
    def __init__(self, audio_path, sr=16000, frame_rate=250, hop_length=512, num_harmonics=32, model_capacity='full'):
        self.audio_path = audio_path
        self.sr = sr
        self.frame_rate = frame_rate
        self.hop_length = int(sr / frame_rate) 
        self.num_harmonics = num_harmonics
        self.model_capacity = model_capacity  # 'tiny', 'full', etc.

    def load_audio(self):
        audio, sr = librosa.load(self.audio_path, sr=self.sr, mono=True)
        return audio, sr

    def compute_f0(self, audio):
        """
        Returns:
          f0_hz:      np.ndarray, shape [n_frames]
          confidence: np.ndarray, shape [n_frames]
        """
        # crepe.predict expects audio in [-1,1], and returns arrays sampled at step_size ms
        step_size = 1000 * self.hop_length / self.sr  # in milliseconds
        # crepe returns (time, frequency, confidence, activation)
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sr=self.sr,
            step_size=step_size,
            model_capacity=self.model_capacity,
            viterbi=True
        )
        return frequency.astype(np.float32), confidence.astype(np.float32)

    def compute_loudness(self, audio):
        """
        Returns:
          loudness_db: np.ndarray, shape [n_frames]
        """
        # Compute frame-level RMS energy
        # hop_length determines frame count = len(audio)/hop_length
        rms = librosa.feature.rms(y=audio, frame_length=self.hop_length*2,
                                  hop_length=self.hop_length, center=True)[0]
        # Convert amplitude → dB
        loudness_db = librosa.amplitude_to_db(rms, ref=1.0)
        return loudness_db.astype(np.float32)

   
    def extract_harmonics(
        self,
        audio: np.ndarray,
        sr: int,
        f0_frames: np.ndarray,
        hop_samples: int,
        num_harmonics: int = 16,
        win_len: int = 8192,
        db_floor: float = -70.0,
        f0_min_hz: float = 40.0,
    ):
        """
        Compute per-frame harmonic magnitudes in dB relative to the fundamental.

        Returns:
            harm_rel_db: [T, H] float32 in [db_floor, 0], with harm_rel_db[:, 0] == 0
            voiced_mask: [T] bool, valid where f0 is voiced and A1>0
        """
        T = int(f0_frames.shape[0])
        H = int(num_harmonics)
        harm_rel_db = np.full((T, H), db_floor, dtype=np.float32)
        voiced_mask = np.zeros((T,), dtype=bool)

        # Frequency grid for this FFT size
        freqs = np.fft.rfftfreq(win_len, d=1.0/sr)
        hann = np.hanning(win_len)
        nyq = sr / 2.0

        for t in range(T):
            f0 = float(f0_frames[t])
            if not np.isfinite(f0) or f0 < f0_min_hz:
                continue

            center = t * hop_samples
            start  = max(0, center - win_len // 2)
            end    = min(audio.shape[0], start + win_len)

            seg = np.zeros((win_len,), dtype=np.float64)
            chunk = audio[start:end].astype(np.float64)
            seg[:chunk.size] = chunk
            seg[:chunk.size] *= hann[:chunk.size]

            mag = np.abs(np.fft.rfft(seg))

            # Harmonic freqs and Nyquist culling
            k = np.arange(1, H+1, dtype=np.float64)
            target_freqs = k * f0
            valid = target_freqs < nyq
            if not np.any(valid):
                continue

            harm_mag = np.interp(target_freqs[valid], freqs, mag)

            A1 = harm_mag[0] if harm_mag.size > 0 else 0.0
            if A1 <= 0.0:
                continue

            eps = 1e-12
            rel_db_valid = 20.0 * np.log10((harm_mag + eps) / (A1 + eps))
            rel_db_valid = np.clip(rel_db_valid, db_floor, 0.0)

            row = np.full((H,), db_floor, dtype=np.float32)
            row[valid] = rel_db_valid.astype(np.float32)
            row[0] = 0.0  # fundamental at 0 dB
            harm_rel_db[t] = row

            voiced_mask[t] = True

        return harm_rel_db, voiced_mask

    def compute_harmonics(self, audio):
        H = []
        f0, conf = self.compute_f0(audio)
        for h in range(1, self.num_harmonics+1):
            # instantaneous phase increment per frame in radians
            H.append(np.sin(2*np.pi * f0 * h * (self.hop_length/self.sr)))
        harmonics = np.stack(H, axis=1)  # [n_frames, num_harmonics]
        return harmonics

    def extract_features(self):
        """
        Load audio and compute frame-rate features aligned to the same T.
        Returns:
            audio:        [N] float32
            f0:           [T] float32 (Hz)
            confidence:   [T] float32
            loudness:     [T] float32 (dB, or whatever your compute_loudness returns)
            harmonics:    [T, H] float32 (e.g., dB rel. H1 or linear ratios)
            voiced_mask:  [T] bool
            hop_length:   int (samples)
            sample_rate:  int (Hz)
            H:            int (num harmonics)
        """
        audio, sr = self.load_audio()                    # keep sr from file
        f0, conf  = self.compute_f0(audio)               # [T_f0], [T_f0]
        loud      = self.compute_loudness(audio)         # [T_loud]

        # If your harmonics extractor uses f0 (recommended), call it with f0 & hop:
        # It should return per-frame [T_harm, H] and a voiced mask [T_harm]
        H = getattr(self, "num_harmonics", 32)
        harm_rel_db, voiced_mask = self.extract_harmonics(
            audio=audio,
            f0_frames=f0,
            hop_samples=self.hop_length,
            sr=sr,
            num_harmonics=H,
        )  # harm: [T_harm, H], voiced_mask: [T_harm]

        # Align all frame-rate arrays to the same T
        T = min(len(f0), len(conf), len(loud), harm_rel_db.shape[0])
        feats = {
            "audio":        audio.astype("float32", copy=False),
            "f0":           f0[:T].astype("float32", copy=False),
            "confidence":   conf[:T].astype("float32", copy=False),
            "loudness":     loud[:T].astype("float32", copy=False),
            "harm_rel_db":    harm_rel_db[:T].astype("float32", copy=False),
            "voiced_mask":  voiced_mask[:T],
            "hop_length":   int(self.hop_length),
            "sample_rate":  int(sr),
            "H":            int(H),
        }
        return feats

