import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from model import *
from data import *

# ==============================
# ⚙️ Config
# ==============================
BATCH_SIZE = 2
EPOCHS = 1
TRAIN_PATH = 'data/membrane/train'
TEST_PATH = 'data/membrane/test'
MODEL_PATH = 'unet_membrane.hdf5'

# Optional: set GPU (commented intentionally)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==============================
# 🔄 Data Augmentation
# ==============================
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = trainGenerator(
    BATCH_SIZE,
    TRAIN_PATH,
    'image',
    'label',
    data_gen_args,
    save_to_dir=None
)

# ==============================
# 🧠 Model
# ==============================
model = unet()

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='loss',
    verbose=1,
    save_best_only=True
)

# ==============================
# 🚨 Intentional Issue Here
# Hardcoded steps_per_epoch
# ==============================
model.fit(
    train_gen,
     steps_per_epoch=len(os.listdir(os.path.join(TRAIN_PATH, 'image'))), 
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# ==============================
# 🧪 Testing
# ==============================
test_gen = testGenerator(TEST_PATH)

results = model.predict(
    test_gen,
    steps=30,
    verbose=1
)

saveResult(TEST_PATH, results)
