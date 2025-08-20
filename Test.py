import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from Data_Process import test_ds

model = load_model(
    'unet_model4.keras',
    custom_objects={'MeanIoU': tf.keras.metrics.MeanIoU}
)

results = model.evaluate(test_ds)
print(dict(zip(model.metrics_names, results)))

for imgs, masks in test_ds.take(1):
    preds = model.predict(imgs)                  
    bin_preds = (preds > 0.5).astype(np.uint8)   

    B = imgs.shape[0]
    for i in range(min(B, 4)):
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(imgs[i]);         axes[0].set_title('Input RGB'); axes[0].axis('off')
        axes[1].imshow(masks[i,:,:,0], cmap='gray')
        axes[1].set_title('Ground Truth');     axes[1].axis('off')
        axes[2].imshow(bin_preds[i,:,:,0], cmap='gray')
        axes[2].set_title('Prediction');       axes[2].axis('off')
        plt.show()
    break