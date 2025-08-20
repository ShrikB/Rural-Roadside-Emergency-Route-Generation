#Unet structure:
#Contracting path - downsampling layers via convolution and sampling
#Expansive path - Upsampling layers via interpolation and convolutions
#Skip connections - connections preserving information from contacting path into expansive path

import tensorflow as tf
from tensorflow.keras import Model, layers, Input

#2 classes in dataset, 2 feature maps
#Convolutions for contracting path
def unet_model(imageL=512, imageW=512, channels=3):
  inputs = Input(shape=(imageL, imageW, channels))
  print(inputs)
  def double_conv(x, filters):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    return x

  def down_sample(x, filters):
    conv = double_conv(x, filters)
    pool = layers.MaxPooling2D(pool_size=2, strides=2)(conv)
    return conv, pool

  def up_sample(x, skip, filters):
    up = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    up = layers.Concatenate(axis=-1)([up, skip])
    up = double_conv(up, filters)
    return up

  down1, pool1 = down_sample(inputs, 64)
  down2, pool2 = down_sample(pool1, 128)
  down3, pool3 = down_sample(pool2, 256)

  bn = double_conv(pool3, 512)

  up3 = up_sample(bn, down3, 256)
  up2 = up_sample(up3, down2, 128)
  up1 = up_sample(up2, down1, 64)

  outp = layers.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(up1)
  model = Model(inputs, outp)
  return model

model = unet_model()
model.summary()
tf.keras.utils.plot_model(
    model,
    to_file="unet_horizontal.png",   
    show_shapes=True,                
    rankdir="LR",                   
    dpi=150                          
)