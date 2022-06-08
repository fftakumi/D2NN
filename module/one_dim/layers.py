import tensorflow as tf
import numpy as np
import math


class IntensityToElectricField(tf.keras.layers.Layer):
    def __init__(self, phi_ini=0.0):
        super(IntensityToElectricField, self).__init__()
        self.phi_ini = phi_ini

    def get_config(self):
        config = super().get_config()
        config.update({
            "phi_ini": self.phi_ini
        })
        return config

    @tf.function
    def call(self, x):
        return tf.complex(tf.sqrt(2 * x), 0.0) * tf.complex(tf.cos(self.phi_ini), tf.sin(self.phi_ini))


class Modulation(tf.keras.layers.Layer):
    def __init__(self, limitation=None, phi_max=0.0):
        super(Modulation, self).__init__()
        if limitation is not None:
            self.limitation = tf.Variable(limitation, validate_shape=False, name="limitation", trainable=False)
            self.limitation = limitation
        else:
            self.limitation = tf.Variable("None", validate_shape=False, name="limitation", trainable=False)
            self.limitation = limitation

        self.phi_max = tf.Variable(phi_max, validate_shape=False, name="phi_max", trainable=False)
        assert self.phi_max.numpy() >= 0.0

    def build(self, input_dim):
        self.input_dim = input_dim
        self.phi = self.add_weight("phi",
                                   shape=[int(input_dim[-1])])
        super(Modulation, self).build(input_dim)

    @tf.function
    def get_limited_phi(self):
        if self.limitation == 'sigmoid':
            return self.phi_max * tf.sigmoid(self.phi)
        else:
            return self.phi

    def get_config(self):
        config = super().get_config()
        config.update({
            "limitation": self.limitation,
            "phi_max": self.phi_max.numpy()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, x):
        phi = self.get_limited_phi()
        return x * tf.complex(tf.cos(phi), tf.sin(phi))


class ElectricFieldToIntensity(tf.keras.layers.Layer):
    def __init__(self, normalization=None):
        super(ElectricFieldToIntensity, self).__init__()
        self.normalization = normalization

    def build(self, input_dim):
        self.input_dim = input_dim
        super(ElectricFieldToIntensity, self).build(input_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, x):
        intensity = tf.abs(x) ** 2 / 2
        if self.normalization == "max":
            intensity = intensity/tf.reduce_max(intensity, axis=1, keepdims=True)
        return intensity


class AngularSpectrum(tf.keras.layers.Layer):
    def __init__(self, wavelength=633.0e-9, z=0.0, d=1.0e-6, n=1.0, method=None):
        super(AngularSpectrum, self).__init__()
        self.wavelength = wavelength
        self.wavelength_effect = wavelength / n
        self.k_effect = 2 * np.pi / self.wavelength_effect
        self.z = z
        self.d = d
        self.n = n
        self.method = method

        assert self.k_effect >= 0.0
        assert self.z >= 0.0
        assert self.d > 0.0
        assert self.n > 0.0

    def build(self, input_dim):
        self.input_dim = input_dim

        width = self.input_dim[-1]
        u = np.fft.fftfreq(width, d=self.d)
        w = np.where(u ** 2 <= 1 / self.wavelength_effect ** 2, tf.sqrt(1 / self.wavelength_effect ** 2 - u ** 2), 0).astype('float64')
        h = np.exp(1.0j * 2 * np.pi * w * self.z)
        if self.method == "band_limited":
            du = 1 / (2 * width * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_effect
            u_filter = np.where(np.abs(u) / (2 * u_limit) <= 1 / 2, 1, 0)
            h = h * u_filter
        elif self.method == "expand":
            self.pad_width = math.ceil(self.input_dim[-1] / 2)
            self.padded_width = int(input_dim[-1] + self.pad_width * 2)

            u = np.fft.fftfreq(self.padded_width, d=self.d)

            du = 1 / (self.padded_width * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_effect

            u_filter = np.where(np.abs(u) <= u_limit, 1.0, 0.0)

            w = np.where(u ** 2 <= 1 / self.wavelength_effect ** 2, tf.sqrt(1 / self.wavelength_effect ** 2 - u ** 2), 0).astype('float64')
            h = np.exp(1.0j * 2 * np.pi * w * self.z)
            h = h * u_filter

        self.res = tf.cast(tf.complex(h.real, h.imag), dtype=tf.complex64)
        super(AngularSpectrum, self).build(input_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "wavelength": self.wavelength,
            "wavelength_effect": self.wavelength_effect,
            "k_effect": self.k_effect,
            "z": self.z,
            "d": self.d,
            "n": self.n
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, x):
        if self.method == "band_limited":
            fft_x = tf.signal.fft(x)
            return tf.signal.ifft(fft_x * self.res)
        elif self.method == 'expand':
            padding = [[0, 0], [self.pad_width, self.pad_width]]
            images_pad = tf.pad(x, paddings=padding)
            images_pad_fft = tf.signal.fft(images_pad)
            u_images_pad = tf.signal.ifft(images_pad_fft * self.res)
            u_images = tf.keras.layers.Lambda(lambda x: x[:, self.pad_width:self.pad_width + self.input_dim[--1]])(u_images_pad)
            return u_images
        else:
            fft_x = tf.signal.fft(x)
            return tf.signal.ifft(fft_x * self.res)


class Detector(tf.keras.layers.Layer):
    def __init__(self, output_dim, padding=0.0, interval=1.0):
        super(Detector, self).__init__()
        self.output_dim = output_dim
        self.padding = padding
        self.interval = interval
        assert 0.0 <= self.padding < 1.0
        assert 0.0 <= self.interval <= 1.0

    def build(self, input_dim):
        ######++++--++++--++++--++++--++++--++++######
        # #:padding
        # +:window
        # -:interval
        self.input_dim = input_dim
        window_width = (1 - self.padding) * self.input_dim[-1] / (self.output_dim + (self.output_dim - 1) * self.interval)
        filters = np.zeros([self.input_dim[-1], self.output_dim])
        for i in range(self.output_dim):
            start = int(self.padding * self.input_dim[-1] / 2 + i * (window_width + window_width * self.interval))
            end = int(start + window_width)
            filters[start:end, i] = 1.0

        self.filters = tf.constant(filters, dtype=tf.float32)
        super(Detector, self).build(input_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "padding": self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, x):
        return tf.tensordot(x, self.filters, axes=[-1, 0])


class Filter(Detector):
    def __init__(self, output_dim, **kwargs):
        super(Filter, self).__init__(output_dim, **kwargs)

    def build(self, input_dim):
        super(Filter, self).build(input_dim)
        self.filter = tf.reduce_sum(self.filters, axis=1)

    @tf.function
    def call(self, x):
        return x * self.filter


class ImageResizing(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(ImageResizing, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, x):
        x_expnad = tf.image.resize(tf.expand_dims(x, -1), self.output_dim)
        x_expnad = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(x_expnad)
        return x_expnad


class ImageTo1D(tf.keras.layers.Layer):
    def __init__(self):
        super(ImageTo1D, self).__init__()

    def build(self, input_dim):
        self.input_dim = input_dim
        super(ImageTo1D, self).build(input_dim)

    @tf.function
    def call(self, x):
        return tf.reshape(x, [-1, self.input_dim[-2] * self.input_dim[-1]])


if __name__ == "__main__":
    pass
