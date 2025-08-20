import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Model import unet_model
from Data_Process import train_ds, val_ds

model = unet_model(imageL=512, imageW=512, channels=3)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=2)
    ]
)

callbacks = [
    ModelCheckpoint('unet_model4.keras', save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
]

#Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,            
    callbacks=callbacks,
    verbose=1
)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
