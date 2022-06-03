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

        self.phi_max = tf.Variable(phi_max, validate_shape=False, name="theta_max", trainable=False)
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
            "phi_max": self.phi_max
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, x):
        return x * tf.complex(tf.cos(self.phi), tf.sin(self.phi))


if __name__ == "__main__":
    pass
