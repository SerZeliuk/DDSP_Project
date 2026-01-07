import tensorflow as tf
import numpy as np
import librosa
import os


class ADSRModel:
    """
    ADSRModel v6 – Pluck-Aware
    Learns envelope shape from note-change features (f0, Δf0, onsets),
    with flattened audio inputs for true amplitude prediction.
    Produces expressive attack/decay envelopes suitable for instruments
    like guitar, bass, violin, etc.
    """

    def __init__(
        self,
        frames_win,
        sample_rate,
        frame_rate,
        batch_size=4,
        lr=1e-4,
        model_dir="./models",
        name="ADSR_model_v6_pluck",
        loss_logger=None,
    ):
        self.frames_win = frames_win
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.batch_size = batch_size
        self.lr = lr
        self.model_dir = model_dir
        self.name = name
        self.loss_logger = loss_logger

        os.makedirs(model_dir, exist_ok=True)
        self.weights_path = os.path.join(model_dir, f"{name}.h5")

        self.model = self._build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=self._hybrid_loss
        )

    # ───────────────────────────────────────────────────────────────
    # Model definition
    # ───────────────────────────────────────────────────────────────
    def _build_model(self, hidden=128):
        """GRU model predicting per-frame amplitude envelope."""
        f0_in    = tf.keras.Input(shape=(self.frames_win, 1), name="f0")
        df0_in   = tf.keras.Input(shape=(self.frames_win, 1), name="df0")
        onset_in = tf.keras.Input(shape=(self.frames_win, 1), name="onset")
        loud_in  = tf.keras.Input(shape=(self.frames_win, 1), name="loudness")

        x = tf.keras.layers.Concatenate()([f0_in, df0_in, onset_in, loud_in])
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                hidden,
                return_sequences=True,
                dropout=0.15,
                recurrent_dropout=0.1,
            )
        )(x)
        x = tf.keras.layers.Dense(hidden, activation="relu")(x)
        x = tf.keras.layers.Dense(hidden // 2, activation="relu")(x)

        # sigmoid -> exponential scaling for decay-like curve
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        env = tf.keras.layers.Lambda(lambda y: tf.pow(y, 1.5))(out)

        return tf.keras.Model(inputs=[f0_in, df0_in, onset_in, loud_in], outputs=env, name=self.name)

    # ───────────────────────────────────────────────────────────────
    # Loss
    # ───────────────────────────────────────────────────────────────
    def _corr_loss(self, y_true, y_pred):
        y_true -= tf.reduce_mean(y_true, axis=1, keepdims=True)
        y_pred -= tf.reduce_mean(y_pred, axis=1, keepdims=True)
        num = tf.reduce_sum(y_true * y_pred, axis=1)
        denom = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=1) *
                        tf.reduce_sum(tf.square(y_pred), axis=1)) + 1e-8
        return 1.0 - tf.reduce_mean(num / denom)

    def _decay_weighted_mse(self, y_true, y_pred, onset_mask):
        base = tf.square(y_true - y_pred)
        weights = tf.clip_by_value(
            tf.nn.avg_pool1d(onset_mask, ksize=7, strides=1, padding="SAME"), 1e-2, 1.0)
        return tf.reduce_mean(base * (1.0 + 2.0 * weights))

    def _hybrid_loss(self, y_true, y_pred):
        if y_true.shape.rank == 2:
            y_true = tf.expand_dims(y_true, -1)
            y_pred = tf.expand_dims(y_pred, -1)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        corr = self._corr_loss(y_true, y_pred)
        d_true = tf.concat([y_true[:, :1, :], y_true[:, 1:, :] - y_true[:, :-1, :]], axis=1)
        onset_mask = tf.cast(tf.abs(d_true) > 0.03, tf.float32)
        decay_mse = self._decay_weighted_mse(y_true, y_pred, onset_mask)
        return 0.2 * mse + 0.6 * decay_mse + 0.2 * corr

    # ───────────────────────────────────────────────────────────────
    # Envelope computation (RMS target)
    # ───────────────────────────────────────────────────────────────
    def _compute_rms_envelope(self, audio_batch):
        envs = []
        for a in audio_batch:
            a = a / (np.sqrt(np.mean(a ** 2)) + 1e-6)  # flatten RMS
            rms = librosa.feature.rms(
                y=a, frame_length=1024, hop_length=self.sample_rate // self.frame_rate
            )[0]
            rms = rms / (np.max(rms) + 1e-6)
            rms = np.power(rms, 0.6)  # emphasize decay
            if len(rms) < self.frames_win:
                rms = np.pad(rms, (0, self.frames_win - len(rms)))
            envs.append(rms[:self.frames_win])
        return np.array(envs, dtype=np.float32)

    # ───────────────────────────────────────────────────────────────
    # Dataset builder
    # ───────────────────────────────────────────────────────────────
    def make_dataset(self, audio_windows, f0_windows, loudness_windows):
        def add_envelope(audio, f0, loud):
            env = tf.py_function(self._compute_rms_envelope, [audio], tf.float32)
            env.set_shape([None, self.frames_win])

            # Flatten inputs per window
            audio_flat = audio / (tf.sqrt(tf.reduce_mean(audio ** 2)) + 1e-6)

            # Frequency & onset features
            f0 = tf.maximum(f0, 1e-6)
            midi = 69.0 + 12.0 * tf.math.log(f0 / 440.0) / tf.math.log(2.0)
            df0 = tf.concat([midi[:, :1], midi[:, 1:] - midi[:, :-1]], axis=1)
            df0_smooth = tf.nn.avg_pool1d(df0[..., None], ksize=5, strides=1, padding="SAME")[..., 0]

            # Normalized loudness (relative only)
            loud_norm = (loud - tf.reduce_mean(loud)) / (tf.math.reduce_std(loud) + 1e-6)
            loud_norm = tf.clip_by_value(loud_norm, -3.0, 3.0)

            onset = tf.cast(tf.abs(df0_smooth) > 0.25, tf.float32)

            inputs = {
                "f0": f0[..., tf.newaxis],
                "df0": df0_smooth[..., tf.newaxis],
                "onset": onset[..., tf.newaxis],
                "loudness": loud_norm[..., tf.newaxis],
            }
            return inputs, env

        ds = tf.data.Dataset.zip((audio_windows, f0_windows, loudness_windows))
        ds = ds.batch(self.batch_size).map(add_envelope, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat().prefetch(tf.data.AUTOTUNE)
        return ds

    # ───────────────────────────────────────────────────────────────
    # Training
    # ───────────────────────────────────────────────────────────────
    def train(self, dataset, num_windows, epochs=80, verbose=1):
        steps_per_epoch = max(1, num_windows // self.batch_size)
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            self.weights_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        callbacks = [ckpt]
        if self.loss_logger is not None:
            callbacks.append(self.loss_logger)
        self.model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            callbacks=callbacks,
        )
        print(f"✅ Training complete. Best weights saved to {self.weights_path}")

    # ───────────────────────────────────────────────────────────────
    # Load / Predict
    # ───────────────────────────────────────────────────────────────
    def load(self):
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print(f"✅ Loaded ADSR weights from {self.weights_path}")
        else:
            print("⚠️ No ADSR weights found, please train first.")

    def predict(self, f0, loud):
        f0 = f0.astype(np.float32)
        f0 = np.maximum(f0, 1e-6)
        midi = 69 + 12 * np.log2(f0 / 440.0)
        df0 = np.concatenate([[0], np.diff(midi)])
        df0_smooth = np.convolve(df0, np.ones(5) / 5, mode="same")
        loud_norm = (loud - np.mean(loud)) / (np.std(loud) + 1e-6)
        loud_norm = np.clip(loud_norm, -3.0, 3.0)
        onset = (np.abs(df0_smooth) > 0.25).astype(np.float32)

        def fix_len(x):
            if len(x) < self.frames_win:
                return np.pad(x, (0, self.frames_win - len(x)))
            return x[:self.frames_win]

        f0, df0_smooth, onset, loud_norm = map(fix_len, [f0, df0_smooth, onset, loud_norm])
        inputs = [f0[None, :, None], df0_smooth[None, :, None], onset[None, :, None], loud_norm[None, :, None]]
        pred = self.model.predict(inputs, verbose=0)[0, :, 0]
        return pred
