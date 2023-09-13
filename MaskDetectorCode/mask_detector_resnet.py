import os
import time
import numpy as np
from tqdm import tqdm
from imutils import paths
from imutils.video import VideoStream

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# import tensorflow as tf
# from keras import layers
# from tensorflow.keras.applications import ResNet50V2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import load_img


# Being forced to do this because for some reason the above throws an error on my laptop
import tensorflow as tf
layers = tf.keras.layers
ResNet50V2 = tf.keras.applications.ResNet50V2
Adam = tf.keras.optimizers.Adam
to_categorical = tf.keras.utils.to_categorical
img_to_array = tf.keras.preprocessing.image.img_to_array
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
load_img = tf.keras.preprocessing.image.load_img

IMG_SIZE = 224
SEP = os.path.sep
#os.path.sep is the '/' for mac and '\' for windows(I think?)

data_dir: str = os.getcwd() + SEP + '..' + SEP + 'MaskData'

print(data_dir)

image_paths = list(paths.list_images(data_dir))

data = []
labels = []

print("[INFO] loading images")
for image_path in tqdm(image_paths, ncols=100, desc='Progress : '):
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = image/255
    data.append(image)

    label = image_path.split(SEP)[-3]
    labels.append(label)
    time.sleep(0.0001)

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

trainX, testX, trainY, testY = train_test_split(data, labels,	test_size=0.20, stratify=labels, random_state=10298)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

feature_extractor_layer = ResNet50V2(weights="imagenet", include_top=False,
	input_tensor = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)))

feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Flatten(name="flatten"),
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dropout(0.5),
    layers.Dense(labels.shape[1], activation='softmax', name='output')
])

model.summary()

LR = 1e-5
EPOCHS = 5
BS = 256

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss="binary_crossentropy" if labels.shape[1] == 2 else "categorical_crossentropy",
  metrics=["accuracy"])

print('[INFO] training the model')
start = time.time()
history = model.fit(aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	epochs=EPOCHS)
print('\nTraining took {}'.format((time.time()-start)))

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save('../MaskDetectorModel/mask_detector_resnet_50_v2.model', save_format="h5")

