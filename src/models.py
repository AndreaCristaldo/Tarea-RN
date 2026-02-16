import tensorflow as tf
from tensorflow.keras import layers, models

def build_encoder(input_dim=784, latent_dim=32):
    inp = layers.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(128, activation="relu", name="encoder_dense_1")(inp)
    x = layers.Dense(64, activation="relu", name="encoder_dense_2")(x)
    z = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    return models.Model(inp, z, name="encoder")

def build_decoder(latent_dim=32, output_dim=784):
    inp = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(64, activation="relu", name="decoder_dense_1")(inp)
    x = layers.Dense(128, activation="relu", name="decoder_dense_2")(x)
    out = layers.Dense(output_dim, activation="sigmoid", name="decoder_output")(x)
    return models.Model(inp, out, name="decoder")

def build_autoencoder(input_dim=784, latent_dim=32):
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)

    ae_inp = layers.Input(shape=(input_dim,), name="autoencoder_input")
    z = encoder(ae_inp)
    recon = decoder(z)

    autoencoder = models.Model(ae_inp, recon, name="autoencoder")
    return autoencoder, encoder, decoder

def build_classifier(encoder: tf.keras.Model, num_classes=10, dropout=0.3):
    inp = layers.Input(shape=encoder.input_shape[1:], name="clf_input")
    z = encoder(inp)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="class")(x)
    return models.Model(inp, out, name="classifier")
