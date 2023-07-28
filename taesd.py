#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf


class Block(tf.keras.layers.Layer):
    def __init__(self, n_out, trainable=True, name=None, **kwargs):
        super(Block, self).__init__(name=name, trainable=trainable, **kwargs)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(n_out, kernel_size=3),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(n_out, kernel_size=3, ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(n_out, kernel_size=3, )], name="conv")
        self.n_out = n_out
        self.fuse = tf.keras.layers.ReLU()

    def build(self, input_shape):
        if input_shape[-1] != self.n_out:
            self.skip = tf.keras.layers.Conv2D(self.n_out, 1, use_bias=False, name="skip")
        else:
            self.skip = lambda x: x
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return self.fuse(self.conv(inputs) + self.skip(inputs))


class Encoder(tf.keras.Sequential):
    def __init__(self, img_height=512, img_width=512):
        super().__init__(
            [
                tf.keras.layers.Input((img_height, img_width, 3), name="encoder_input"),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3),
                Block(64),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, use_bias=False),
                Block(64),
                Block(64),
                Block(64),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, use_bias=False),
                Block(64),
                Block(64),
                Block(64),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, use_bias=False),
                Block(64),
                Block(64),
                Block(64),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(4, kernel_size=3), ]
        )
        weights_fpath = "./taesd_encoder.h5"
        if os.path.exists(weights_fpath):
            self.load_weights(weights_fpath)
            print("loaded :[{}]".format(os.path.basename(weights_fpath)))


class Decoder(tf.keras.Sequential):

    def nearest_neighbor_upsampling(self, input_tensor, scale=None, height_scale=None,
                                    width_scale=None):
        if not scale and (height_scale is None or width_scale is None):
            raise ValueError('Provide either `scale` or `height_scale` and'
                             ' `width_scale`.')
        h_scale = scale if height_scale is None else height_scale
        w_scale = scale if width_scale is None else width_scale
        if h_scale == 1 and w_scale == 1:
            return input_tensor
        height, width, channels = input_tensor.shape.as_list()[1:]
        output_tensor = tf.stack([input_tensor] * w_scale, axis=3)
        output_tensor = tf.stack([output_tensor] * h_scale, axis=2)
        return tf.reshape(output_tensor,
                          [-1, height * h_scale, width * w_scale, channels])

    def __init__(self, img_height=64, img_width=64, name=None):
        super().__init__(
            [
                tf.keras.layers.Input((img_height, img_width, 4), name="decoder_input"),
                tf.keras.layers.Lambda(lambda x: tf.math.tanh(x / 3.0) * 3.0),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3),
                tf.keras.layers.ReLU(),
                Block(64),
                Block(64),
                Block(64),
                tf.keras.layers.Lambda(lambda x: self.nearest_neighbor_upsampling(x, 2)),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False),
                Block(64),
                Block(64),
                Block(64),
                tf.keras.layers.Lambda(lambda x: self.nearest_neighbor_upsampling(x, 2)),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False),
                Block(64),
                Block(64),
                Block(64),
                tf.keras.layers.Lambda(lambda x: self.nearest_neighbor_upsampling(x, 2)),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False),
                Block(64),
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(3, kernel_size=3),
            ],
            name=name,
        )
        weights_fpath = "./taesd_decoder.h5"
        if os.path.exists(weights_fpath):
            self.load_weights(weights_fpath)
            print("loaded :[{}]".format(os.path.basename(weights_fpath)))


class TAESD(object):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.h5", decoder_path="taesd_decoder.h5"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_weights(encoder_path)
        if decoder_path is not None:
            self.decoder.load_weights(decoder_path)

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return np.clip(x / (2.0 * TAESD.latent_magnitude) + TAESD.latent_shift, 0., 1.)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return (x - TAESD.latent_shift) * (2.0 * TAESD.latent_magnitude)


def main():
    from PIL import Image
    import sys
    taesd = TAESD()
    for im_path in sys.argv[1:]:
        im_filename = im_path[:-4]
        im = (np.expand_dims(Image.open(im_path).convert("RGB"), 0).astype("float32")) / 255.0
        # encode image, quantize, and save to file
        im_enc = ((taesd.scale_latents(taesd.encoder(im))) * 255.0).astype("uint8")
        enc_path = im_filename + ".encoded.png"
        Image.fromarray(im_enc[0]).save(enc_path)
        print(f"Encoded {im_path} to {enc_path}")

        # load the saved file, dequantize, and decode
        im = (np.expand_dims(Image.open(enc_path), 0).astype("float32")) / 255.0
        im_enc = taesd.unscale_latents(im)
        im_dec = np.clip(taesd.decoder(im_enc) * 255.0, 0.0, 255.0).astype("uint8")
        dec_path = im_filename + ".decoded.png"
        print(f"Decoded {enc_path} to {dec_path}")
        Image.fromarray(im_dec[0]).save(dec_path)


if __name__ == "__main__":
    main()
