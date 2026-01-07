# ddsp_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from synths import HarmonicPlusNoiseSynth

##############################################################################
# 1. Safe resampling using tf.image.resize
##############################################################################

def safe_resample(x, target_len):
    """
    x: [B, T, C]
    Returns: [B, target_len, C]
    """
    x = tf.cast(x, tf.float32)

    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = tf.shape(x)[2]

    # tf.image.resize expects NHWC
    x4 = tf.reshape(x, [B, T, 1, C])
    x4 = tf.image.resize(x4, [target_len, 1], method="bilinear")
    out = tf.reshape(x4, [B, target_len, C])

    # Replace any NaN/Inf with 0
    out = tf.where(tf.math.is_finite(out), out, tf.zeros_like(out))
    return out


##############################################################################
# 2. Safe StyleVector layer
##############################################################################

class StyleVector(layers.Layer):
    """Trainable style latent with safe initialization."""
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.style = self.add_weight(
            name="instrument_style",
            shape=(self.latent_dim,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        s = tf.repeat(self.style[None, :], B, axis=0)
        # safety: remove NaN from style vector
        s = tf.where(tf.math.is_finite(s), s, tf.zeros_like(s))
        return x+s


##############################################################################
# 3. Full DDSP Autoencoder (Safe Version)
##############################################################################

def build_ddsp_autoencoder(
    sample_rate: int = 16000,
    frame_rate:   int = 250,
    window_sec:   int = 4,
    conv_channels:   int = 64,
    num_layers:      int = 4,
    kernel_size:     int = 3,
    latent_dim:      int = 128,
    decoder_hidden:  int = 256,
    n_harmonics:     int = 64,
    synth_window:    int = 257
) -> tf.keras.Model:

    SR = sample_rate
    FR = frame_rate
    HOP = SR // FR
    SAMPLES_WIN = window_sec * SR
    FRAMES_WIN = window_sec * FR

    ##########################################################################
    # 1) Inputs
    ##########################################################################

    audio_in = tf.keras.Input(shape=(SAMPLES_WIN,),   name="audio")
    f0_in    = tf.keras.Input(shape=(FRAMES_WIN, 1),  name="f0")
    loud_in  = tf.keras.Input(shape=(FRAMES_WIN, 1),  name="loudness")

    ##########################################################################
    # 2) Encoder
    ##########################################################################

    x = layers.Reshape((SAMPLES_WIN, 1), name="reshape_audio")(audio_in)

    for i in range(num_layers):
        x = layers.Conv1D(
            conv_channels,
            kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"enc_conv_{i}",
        )(x)

    z = layers.GlobalAveragePooling1D(name="enc_pool")(x)
    z = layers.Dense(latent_dim, kernel_initializer="he_normal", name="enc_fc")(z)

    # Style vector
    z = StyleVector(latent_dim, name="instrument_style_vector")(z)

    # Stability
    z = layers.LayerNormalization(name="latent_norm")(z)


    ##########################################################################
    # 3) Decoder
    ##########################################################################

    # Tile latent to frame-rate
    z_t = layers.Lambda(
        lambda t: tf.repeat(tf.expand_dims(t, 1), FRAMES_WIN, axis=1),
        name="tile_z"
    )(z)

    controls = layers.Concatenate(name="concat_controls")([f0_in, loud_in])
    d = layers.Concatenate(name="decoder_input")([z_t, controls])

    # Normalize to prevent explosions
    d = layers.LayerNormalization(name="decoder_in_norm")(d)

    # First decoder FC
    d = layers.Lambda(lambda x: tf.clip_by_value(x, -20.0, 20.0))(d)
    d = layers.Dense(
        decoder_hidden,
        activation="relu",
        kernel_initializer="he_normal",
        name="dec_fc1"
    )(d)

    # Clip for safety before harmonic prediction
    d_harm = layers.Lambda(lambda x: tf.clip_by_value(x, -15.0, 15.0))(d)

    # Harmonics at frame-rate
    harm = layers.Dense(
        n_harmonics,
        kernel_initializer="glorot_uniform",
        name="harm_fc"
    )(d_harm)

    # Clip logits before softmax
    harm = layers.Lambda(lambda x: tf.clip_by_value(x, -20.0, 20.0))(harm)

    harm = layers.Softmax(name="harmonic_distribution")(harm)

    # Remove any NaN from harmonics (hard safety)
    harm = layers.Lambda(lambda x: tf.where(tf.math.is_finite(x), x, tf.zeros_like(x)),
                         name="harm_nan_guard")(harm)

    # Upsample harmonics to sample-rate (safe)
    harm_up = layers.Lambda(
        lambda h: safe_resample(h, SAMPLES_WIN),
        name="resample_harmonics"
    )(harm)

    ##########################################################################
    # Convert decoder frame-rate features to sample-rate features
    ##########################################################################

    d2 = layers.Dense(
        decoder_hidden * HOP,
        activation="relu",
        kernel_initializer="he_normal",
        name="dec_fc2"
    )(d)

    d2 = layers.Reshape((SAMPLES_WIN, decoder_hidden), name="dec_reshape")(d2)


    ##########################################################################
    # 4) DSP Param Projections
    ##########################################################################

    amp = layers.Conv1D(
        1,
        1,
        activation="relu",
        kernel_initializer="he_normal",
        name="amplitude"
    )(d2)

    noise = layers.Conv1D(
        1,
        1,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="noise_magnitudes"
    )(d2)

    # Resample f0 safely
    f0_up = layers.Lambda(
        lambda f: safe_resample(f, SAMPLES_WIN),
        name="resample_f0"
    )(f0_in)

    ##########################################################################
    # 5) Synthesizer
    ##########################################################################

    synth = HarmonicPlusNoiseSynth(
        n_samples=SAMPLES_WIN,
        sample_rate=SR,
        n_harmonics=n_harmonics,
        window_size=synth_window,
    )

    recon = synth(amp, harm_up, noise, f0_up)

    ##########################################################################
    # 6) Body Filter
    ##########################################################################

    x = layers.Reshape((SAMPLES_WIN, 1), name="body_in")(recon)

    body = layers.Conv1D(
        filters=1,
        kernel_size=1024,
        padding='same',
        use_bias=False,
        kernel_initializer="he_normal",
        name='body_filter'
    )(x)

    recon = layers.Reshape((SAMPLES_WIN,), name="body_out")(body)

    # Final clamp to [-1, 1]
    recon = layers.Lambda(
        lambda x: tf.clip_by_value(x, -1.0, 1.0),
        name="clip_output"
    )(recon)

    ##########################################################################
    # 7) Build model
    ##########################################################################

    model = Model(
        inputs=[audio_in, f0_in, loud_in],
        outputs={"recon": recon, "harm_pred": harm},
        name="ddsp_autoencoder_safe"
    )

    return model
