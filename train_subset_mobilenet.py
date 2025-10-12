# train_subset_mobilenet.py
import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks
from pathlib import Path

# Config
DATA_DIR = "dataset_subset"
IMG_SIZE = (160,160)    # MobileNetV2 works well with 160 or 224
BATCH_SIZE = 32
EPOCHS = 12
MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(exist_ok=True)

# Data generators
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                               rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(os.path.join(DATA_DIR, "train"),
                                         target_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE,
                                         class_mode="categorical")

val_ds = val_gen.flow_from_directory(os.path.join(DATA_DIR, "val"),
                                    target_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE,
                                    class_mode="categorical",
                                    shuffle=False)

num_classes = len(train_ds.class_indices)
print("Classes:", train_ds.class_indices)

# Build model (Transfer Learning)
base = MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet", pooling="avg")
base.trainable = False  # freeze base

inputs = layers.Input(shape=(*IMG_SIZE,3))
x = base(inputs, training=False)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Callbacks
ckpt = callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "asl_subset_mobilenet.h5"),
                                 monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Train
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[ckpt, reduce_lr, early])

# Save class indices and history plot data
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_ds.class_indices, f)

print("Training finished. Best model saved to models/asl_subset_mobilenet.h5")
