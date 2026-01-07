from typing import Dict, Text
import tensorflow as tf
import core_functions as core
tfkl = tf.keras.layers

# Define Types.
TensorDict = Dict[Text, tf.Tensor]

class Processor(tfkl.Layer):
  """Abstract base class for signal processors.

  Since most effects / synths require specificly formatted control signals
  (such as amplitudes and frequenices), each processor implements a
  get_controls(inputs) method, where inputs are a variable number of tensor
  arguments that are typically neural network outputs. Check each child class
  for the class-specific arguments it expects. This gives a dictionary of
  controls that can then be passed to get_signal(controls). The
  get_outputs(inputs) method calls both in succession and returns a nested
  output dictionary with all controls and signals.
  """

  def __init__(self, name: Text, trainable: bool = False):
    super().__init__(name=name, trainable=trainable, autocast=False)

  def call(self,
           *args: tf.Tensor,
           return_outputs_dict: bool = False,
           **kwargs) -> tf.Tensor:
    """Convert input tensors arguments into a signal tensor."""
    # Don't use `training` or `mask` arguments from keras.Layer.
    for k in ['training', 'mask']:
      if k in kwargs:
        _ = kwargs.pop(k)

    controls = self.get_controls(*args, **kwargs)
    signal = self.get_signal(**controls)
    if return_outputs_dict:
      return dict(signal=signal, controls=controls)
    else:
      return signal

  def get_controls(self, *args: tf.Tensor, **kwargs: tf.Tensor) -> TensorDict:
    """Convert input tensor arguments into a dict of processor controls."""
    raise NotImplementedError

  def get_signal(self, *args: tf.Tensor, **kwargs: tf.Tensor) -> tf.Tensor:
    """Convert control tensors into a signal tensor."""
    raise NotImplementedError

class HarmonicPlusNoiseSynth(Processor):
    """
    Custom synthesizer combining harmonic sinusoidal synthesis and filtered noise.
    Integrates seamlessly with DDSPAutoencoder.
    """
    def __init__(self,
                 n_samples: int,
                 sample_rate: int,
                 n_harmonics: int,
                 window_size: int = 257,
                 name: str = 'custom_harmonic_plus_noise'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.window_size = window_size

    def get_controls(self,
                     amplitudes: tf.Tensor,
                     harmonic_distribution: tf.Tensor,
                     noise_magnitudes: tf.Tensor,
                     f0_hz: tf.Tensor) -> dict:
        """
        Scale and normalize network outputs into synthesizer controls.

        Args:
          amplitudes: [batch, n_frames, 1]
          harmonic_distribution: [batch, n_frames, n_harmonics]
          noise_magnitudes: [batch, n_frames, 1]
          f0_hz: [batch, n_frames, 1]

        Returns:
          Dict with keys:
            'amplitudes', 'harmonic_distribution', 'noise_magnitudes', 'f0_hz'
        """
        # Scale amplitude and noise via exp_sigmoid for positivity
        amps = core.exp_sigmoid(amplitudes)
        noise = core.exp_sigmoid(noise_magnitudes)
        # noise = tf.zeros_like(noise)
        n_bins = self.window_size // 2 + 1
        noise = tf.tile(noise, [1, 1, n_bins])
        # Normalize harmonic distribution below Nyquist
        harm_dist = core.normalize_harmonics(harmonic_distribution, f0_hz, self.sample_rate)

        return {
            'amplitudes': amps,
            'harmonic_distribution': harm_dist,
            'noise_magnitudes': noise,
            'f0_hz': f0_hz
        }

    def get_signal(self,
                   amplitudes: tf.Tensor,
                   harmonic_distribution: tf.Tensor,
                   noise_magnitudes: tf.Tensor,
                   f0_hz: tf.Tensor) -> tf.Tensor:
        """
        Produce audio by summing harmonic and filtered noise branches.

        Args:
          amplitudes: [batch, n_frames, 1]
          harmonic_distribution: [batch, n_frames, n_harmonics]
          noise_magnitudes: [batch, n_frames, 1]
          f0_hz: [batch, n_frames, 1]

        Returns:
          audio: [batch, n_samples]
        """
        # Harmonic branch
        harmonic_signal = core.harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate
        )

        # Noise branch
        # batch_size = tf.shape(noise_magnitudes)[0]
        # White noise in [-1,1]
        # noise = tf.random.uniform([batch_size, self.n_samples], minval=-1.0, maxval=1.0)
        # filtered_noise = core.frequency_filter(
        #     noise,
        #     noise_magnitudes,
        #     window_size=self.window_size
        # )

        # Mix branches
        audio = harmonic_signal #+ 0.001*filtered_noise
        return audio
