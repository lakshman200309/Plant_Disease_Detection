import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'PlantVillage')
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'val')
DISEASE_INFO_PATH = os.path.join(BASE_DIR, 'disease_info.json')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.h5')
CLASS_INDEX_PATH = os.path.join(BASE_DIR, 'class_indices.json')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model definition
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Save class indices with consistent key types (str keys for JSON compatibility)
class_indices = train_generator.class_indices
disease_info = {}

# Load disease info
with open(DISEASE_INFO_PATH, 'r', encoding='utf-8') as f:
    disease_info_data = json.load(f)

# Build final class index info
for disease, index in class_indices.items():
    disease_info[str(index)] = {
        "name": disease,
        "cure": disease_info_data.get(disease, {}).get("cure", "No data"),
        "growth_tips": disease_info_data.get(disease, {}).get("growth_tips", "No data")
    }

# Save class index mapping
with open(CLASS_INDEX_PATH, 'w', encoding='utf-8') as f:
    json.dump(disease_info, f, indent=4, ensure_ascii=False)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Save the model
model.save(MODEL_SAVE_PATH)

print("✅ Training complete. Model saved to", MODEL_SAVE_PATH)
print("✅ Class indices saved to", CLASS_INDEX_PATH)
