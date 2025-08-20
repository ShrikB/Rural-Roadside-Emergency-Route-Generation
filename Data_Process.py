import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt

base_dir = 'deepglobe/'
train_dir = os.path.join(base_dir, 'train/')
test_dir =  os.path.join(base_dir, 'testset/')
img_size = (512, 512)
batch_size = 4

img_paths  = sorted(glob(os.path.join(train_dir, '*_sat.jpg')))
mask_paths = sorted(glob(os.path.join(train_dir, '*_mask.png')))

test_imgs = sorted(glob(os.path.join(test_dir, '*_sat.jpg')))
test_masks = sorted(glob(os.path.join(test_dir, '*_mask.png')))

img_exp = load_img(train_dir + '655_sat.jpg', target_size=(512,512))
pth_exp = load_img(train_dir + '655_mask.png', target_size=(512,512))
arr = img_to_array(img_exp)
print(arr.shape)

plt.figure(figsize=(6,6))
plt.imshow(img_exp)
plt.axis('off')
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(pth_exp)
plt.axis('off')  
plt.show()

#Split into train/validation lists
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    img_paths, mask_paths,
    test_size=0.20,
    random_state=49,
    shuffle=True
)

#Preprocessing function
def _parse_pair(img_path, mask_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)


    img  = tf.image.resize(img, img_size, method='area')
    mask = tf.image.resize(mask, img_size, method='nearest')

 
    img  = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask > 128, tf.uint8)

    return img, mask


def make_dataset(img_list, mask_list, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((img_list, mask_list))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_list))
    ds = ds.map(_parse_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_imgs, train_masks, shuffle=True)
val_ds   = make_dataset(val_imgs,   val_masks,   shuffle=False)
test_ds  = make_dataset(test_imgs,  test_masks,  shuffle=False)
# 5) Quick sanity check
for imgs, msks in train_ds.take(1):
    print("Batch img shape:", imgs.shape)   # → (batch_size, 256,256,3)
    print("Batch mask shape:", msks.shape)  # → (batch_size, 256,256,1)
