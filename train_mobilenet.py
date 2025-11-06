# train_mobilenet_folder.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks, optimizers

# Paths
train_dir = "dataset/train"
val_dir = "dataset/test"
MODEL_PATH = "models/mobilenet_emotion.h5"

# Parameters
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = len(os.listdir(train_dir))

print("📥 Loading dataset from folders...")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("✅ Dataset ready.")

# Build MobileNetV2
print("⚙️ Building MobileNetV2 model...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(optimizer=optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Callbacks
cb = [
    callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

print("🚀 Training...")
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=cb)

print("🔍 Fine-tuning top layers...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_gen,
          validation_data=val_gen,
          epochs=5,
          callbacks=cb)

print("💾 Saving model...")
model.save(MODEL_PATH)
print("✅ Training complete! Model saved at", MODEL_PATH)
