import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.losses import MeanSquaredError
# from ddsp.losses import SpectralLoss
import core_functions as core
import functools
import spectral_operations as spectral_ops

mse = MeanSquaredError()


def mean_difference(target, value, loss_type='L1', weights=None):
  """Common loss functions.

  Args:
    target: Target tensor.
    value: Value tensor.
    loss_type: One of 'L1', 'L2', or 'COSINE'.
    weights: A weighting mask for the per-element differences.

  Returns:
    The average loss.

  Raises:
    ValueError: If loss_type is not an allowed value.
  """
  difference = target - value
  weights = 1.0 if weights is None else weights
  loss_type = loss_type.upper()
  if loss_type == 'L1':
    return tf.reduce_mean(tf.abs(difference * weights))
  elif loss_type == 'L2':
    return tf.reduce_mean(difference**2 * weights)
  elif loss_type == 'COSINE':
    return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
  else:
    raise ValueError('Loss type ({}), must be '
                     '"L1", "L2", or "COSINE"'.format(loss_type))

class Loss(tfkl.Layer):
  """Base class. Duck typing: Losses just must implement get_losses_dict()."""

  def get_losses_dict(self, *args, **kwargs):
    """Returns a dictionary of losses for the model."""
    loss = self(*args, **kwargs)
    return {self.name: loss}

class SpectralLoss(Loss):
  """Multi-scale spectrogram loss.

  This loss is the bread-and-butter of comparing two audio signals. It offers
  a range of options to compare spectrograms, many of which are redunant, but
  emphasize different aspects of the signal. By far, the most common comparisons
  are magnitudes (mag_weight) and log magnitudes (logmag_weight).
  """

  def __init__(self,
               fft_sizes=(2048, 1024, 512, 256, 128, 64),
               loss_type='L1',
               mag_weight=1.0,
               delta_time_weight=0.0,
               delta_freq_weight=0.0,
               cumsum_freq_weight=0.0,
               logmag_weight=0.0,
               loudness_weight=0.0,
               name='spectral_loss'):
    """Constructor, set loss weights of various components.

    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.fft_sizes = fft_sizes
    self.loss_type = loss_type
    self.mag_weight = mag_weight
    self.delta_time_weight = delta_time_weight
    self.delta_freq_weight = delta_freq_weight
    self.cumsum_freq_weight = cumsum_freq_weight
    self.logmag_weight = logmag_weight
    self.loudness_weight = loudness_weight

    self.spectrogram_ops = []
    for size in self.fft_sizes:
      spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size)
      self.spectrogram_ops.append(spectrogram_op)

  def call(self, target_audio, audio, weights=None):
    loss = 0.0

    diff = core.diff
    cumsum = tf.math.cumsum

    # Compute loss for each fft size.
    for loss_op in self.spectrogram_ops:
      target_mag = loss_op(target_audio)
      value_mag = loss_op(audio)

      # Add magnitude loss.
      if self.mag_weight > 0:
        loss += self.mag_weight * mean_difference(
            target_mag, value_mag, self.loss_type, weights=weights)

      if self.delta_time_weight > 0:
        target = diff(target_mag, axis=1)
        value = diff(value_mag, axis=1)
        loss += self.delta_time_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      if self.delta_freq_weight > 0:
        target = diff(target_mag, axis=2)
        value = diff(value_mag, axis=2)
        loss += self.delta_freq_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      # TODO(kyriacos) normalize cumulative spectrogram
      if self.cumsum_freq_weight > 0:
        target = cumsum(target_mag, axis=2)
        value = cumsum(value_mag, axis=2)
        loss += self.cumsum_freq_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      # Add logmagnitude loss, reusing spectrogram.
      if self.logmag_weight > 0:
        target = spectral_ops.safe_log(target_mag)
        value = spectral_ops.safe_log(value_mag)
        loss += self.logmag_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

    if self.loudness_weight > 0:
      target = spectral_ops.compute_loudness(target_audio, n_fft=2048,
                                             use_tf=True)
      value = spectral_ops.compute_loudness(audio, n_fft=2048, use_tf=True)
      loss += self.loudness_weight * mean_difference(
          target, value, self.loss_type, weights=weights)

    return core.nan_to_num(loss, 0.0)



# Multi‐scale spectral loss via ddsp’s SpectralLoss layer
spectral_loss = SpectralLoss(
    fft_sizes=(2048, 1024, 512, 256),
    loss_type='L1',
    mag_weight=1.0,
    logmag_weight=1.0,
    delta_time_weight=0.0,
    delta_freq_weight=0.0,
    cumsum_freq_weight=0.0,
    loudness_weight=0.0,
)

def combined_loss(y_true, y_pred, alpha=0.1):
    """
    y_true: [B, T] waveform
    y_pred: [B, T, 2] where
            y_pred[...,0] = f0_aux,
            y_pred[...,1] = audio
    alpha: weight of the pitch‐aux loss
    """
    # split WaveNet outputs
    f0_pred = y_pred[..., 0]
    audio_pred = y_pred[..., 1]
    # waveform spectral loss
    loss_spec = spectral_loss(y_true, audio_pred)
    # pitch MSE
    # here we assume you have f0 targets baked into your features array
    # as column 0 of feats, captured as a Python constant
    # (alternatively you can recompute f0 inside the loss if you pass it in)
    f0_target = tf.constant(feats[:, 0], dtype=tf.float32)[tf.newaxis, :]
    loss_pitch = mse(f0_target, f0_pred)
    return loss_spec + alpha * loss_pitch
