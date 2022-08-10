import tensorflow as tf
import numpy as np
import math


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

    def call(self, x):
        x_expnad = tf.image.resize(tf.expand_dims(x, -1), self.output_dim)
        x_expnad = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(x_expnad)
        return x_expnad


class ImageBinarization(tf.keras.layers.Layer):
    def __init__(self, threshold=0.5, minimum=0.0, maximum=1.0):
        super(ImageBinarization, self).__init__()
        self.threshold = threshold
        self.minimum = minimum
        self.maximum = maximum

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "minimum": self.minimum,
            "maximum": self.maximum
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        return tf.where(x >= self.threshold, self.maximum, self.minimum)


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
        assert limitation == 'sin' or limitation == 'sigmoid'
        self.limitation = limitation if limitation is not None else "None"

        self.phi_max = tf.constant(phi_max, dtype=tf.float32, name="phi_max")
        assert self.phi_max.numpy() >= 0.0

    def build(self, input_dim):
        self.input_dim = input_dim
        self.phi = self.add_weight("phi",
                                   shape=[int(input_dim[-1]), int(input_dim[-2])])
        super(Modulation, self).build(input_dim)

    @tf.function
    def get_limited_phi(self):
        if self.limitation == 'sigmoid':
            return self.phi_max * tf.sigmoid(self.phi)
        elif self.limitation == 'sin':
            return self.phi_max * tf.sin(self.phi) + self.phi_max/2.0
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
    def __init__(self, output_dim, wavelength=633e-9, z=0.0, d=1.0e-6, n=1.0, normalization=None, method=None):
        super(AngularSpectrum, self).__init__()
        self.output_dim = output_dim
        # self.wavelength = wavelength / n
        # self.k = 2 * np.pi / self.wavelength
        # self.z = z
        # self.d = d
        # self.n = n
        self.wavelength = wavelength
        self.wavelength_eff = wavelength / n
        self.k = 2 * np.pi / self.wavelength_eff
        self.z = z
        self.d = d
        self.n = n
        self.normalization = normalization if normalization is not None else "None"
        self.method = method if method is not None else "None"

        assert self.k >= 0.0
        assert self.z >= 0.0
        assert self.d > 0.0
        assert self.n > 0.0

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "wavelength": self.wavelength,
            "k": self.k,
            "z": self.z,
            "d": self.d,
            "n": self.n,
            "normalization": self.normalization,
            "method": self.method
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_dim):
        self.input_dim = input_dim

        width = self.input_dim[-1]
        height = self.input_dim[-2]
        u = np.fft.fftfreq(width, d=self.d)
        v = np.fft.fftfreq(height, d=self.d)
        UU, VV = np.meshgrid(u, v)
        w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength_eff ** 2, tf.sqrt(1 / self.wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
        h = np.exp(1.0j * 2 * np.pi * w * self.z)

        if self.method == 'band_limited':
            du = 1 / (2 * width * self.d)
            dv = 1 / (2 * height * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_eff
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength_eff
            u_filter = np.where(np.abs(UU) / (2 * u_limit) <= 1 / 2, 1, 0)
            v_filter = np.where(np.abs(VV) / (2 * v_limit) <= 1 / 2, 1, 0)
            h = h * u_filter * v_filter
        elif self.method == 'expand':
            self.pad_upper = math.ceil(self.input_dim[-2] / 2)
            self.pad_left = math.ceil(self.input_dim[-1] / 2)
            self.padded_width = int(input_dim[-1] + self.pad_left * 2)
            self.padded_height = int(input_dim[-2] + self.pad_upper * 2)

            u = np.fft.fftfreq(self.padded_width, d=self.d)
            v = np.fft.fftfreq(self.padded_height, d=self.d)

            du = 1 / (self.padded_width * self.d)
            dv = 1 / (self.padded_height * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_eff
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength_eff
            UU, VV = np.meshgrid(u, v)

            u_filter = np.where(np.abs(UU) <= u_limit, 1, 0)
            v_filter = np.where(np.abs(VV) <= v_limit, 1, 0)

            w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength_eff ** 2, tf.sqrt(1 / self.wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
            h = np.exp(1.0j * 2 * np.pi * w * self.z)
            h = h * u_filter * v_filter

        self.res = tf.cast(tf.complex(h.real, h.imag), dtype=tf.complex64)

    @tf.function
    def propagation(self, cximages):
        if self.method == 'band_limited':
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)
        elif self.method == 'expand':
            padding = [[0, 0], [self.pad_upper, self.pad_upper], [self.pad_left, self.pad_left]]
            images_pad = tf.pad(cximages, paddings=padding)
            images_pad_fft = tf.signal.fft2d(images_pad)
            u_images_pad = tf.signal.ifft2d(images_pad_fft * self.res)
            u_images = tf.keras.layers.Lambda(lambda x: x[:, self.pad_upper:self.pad_upper + self.input_dim[-2], self.pad_left:self.pad_left + self.input_dim[-1]])(u_images_pad)
            return u_images
        else:
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)

    @tf.function
    def call(self, x):
        prop_x = self.propagation(x)

        if self.normalization == 'max':
            maximum = tf.reduce_max(tf.abs(prop_x))
            prop_x = prop_x / tf.complex(maximum, 0.0 * maximum)

        return prop_x


class MNISTDetector(tf.keras.layers.Layer):
    def __init__(self, output_dim, inverse=False, activation=None, normalization=None, **kwargs):
        super(MNISTDetector, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.inverse = inverse
        self.activation = activation
        self.normalization = normalization

    @tf.function
    def get_photo_mask(self):
        return tf.reduce_sum(self.filter, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "inverse": self.inverse,
            "activation": self.activation,
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.input_dim = input_shape
        width = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))
        height = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))

        w0 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w0[2 * height:3 * height, width:2 * width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w1[2 * height:3 * height, 4 * width:5 * width] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w2[2 * height:3 * height, 7 * width:8 * width] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w3[4 * height:5 * height, 1 * width:2 * width] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w4[4 * height:5 * height, 3 * width:4 * width] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w5[4 * height:5 * height, 5 * width:6 * width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w6[4 * height:5 * height, 7 * width:8 * width] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w7[6 * height:7 * height, width:2 * width] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w8[6 * height:7 * height, 4 * width:5 * width] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w9[6 * height:7 * height, 7 * width:8 * width] = 1.0
        w9 = tf.constant(w9)

        if self.inverse:
            self.filter = -tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)
        else:
            self.filter = tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    def call(self, x, **kwargs):
        y = tf.tensordot(x, self.filter, axes=[[1, 2], [0, 1]])

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y
