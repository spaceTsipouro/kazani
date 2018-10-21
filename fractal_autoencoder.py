import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from create_dataset import read as read_dataset
from tensorflow.keras import backend as K

layers = keras.layers

BATCH_SIZE = 128
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
latent_dim = 32
epsilon_std = 1 / np.sqrt(2)

def build_encoder(im_input):
  x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(im_input)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
  return encoded

def build_decoder(decoder_input):
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_input)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(16, (3, 3), activation='relu')(x)
  x = layers.UpSampling2D((2, 2))(x)
  decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
  return decoded

def sampling(args):
  z_mean, z_log_var = args
  batch = K.shape(z_mean)[0]
  dims = K.int_shape(z_mean)[1:]
  # by default, random_normal has mean=0 and std=1.0
  epsilon = K.random_normal(shape=dims)
  return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_model():
  im_input = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))
  encoder_output = build_encoder(im_input)
  z_mean = layers.Dense(latent_dim)(encoder_output)
  z_log_sigma = layers.Dense(latent_dim)(encoder_output)
  z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

  decoder_output = build_decoder(z)
  model = keras.models.Model(im_input, decoder_output)
  
  def vae_loss(x, x_decoded_mean):
    xent_loss = K.mean(keras.losses.binary_crossentropy(x, x_decoded_mean), axis=[-2,-1])
    kl_loss = K.mean(- 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1), axis=[-2,-1])
    return xent_loss + kl_loss

  model.compile(optimizer='rmsprop',
                loss=vae_loss,
                metrics=['mae']) # mean absolute error
  return model

def train_model(model, train_data):
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint("./training_weights.hdf5", monitor='loss', period=1)]

  model.fit(train_data, train_data, 
            epochs=50, batch_size=BATCH_SIZE,
            shuffle=True, callbacks=callbacks)

def predict_with_model(model, input_data):
  pr_input = np.reshape(input_data, (len(input_data), IMAGE_HEIGHT, IMAGE_WIDTH, 1)) 
  prediction = model.predict(pr_input)
  return prediction

def main(_):
  train_data = read_dataset(256)
  train_data = train_data.astype('float32') / 255
  train_data = np.reshape(train_data, (len(train_data), IMAGE_HEIGHT, IMAGE_WIDTH, 1))  # adapt this if using `channels_first` image data format
  model = build_model()  
  train_model(model, train_data)

if __name__ == "__main__":
  tf.app.run()