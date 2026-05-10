"""
DEPRECATED — kept for offline / no-internet fallback only.

For the actively maintained training workflow, use the Colab notebook at
`leafdoc-backend/colab/train_on_colab.ipynb`. That notebook trains BOTH the
disease classifier AND the leaf-vs-not-leaf gate that the FastAPI backend
needs at inference time. This script only trains the disease classifier and
will NOT produce the leaf gate.

If you absolutely must train locally, you have been warned: the PlantVillage
dataset is large and MobileNetV3 training will saturate RAM and CPU.
"""

import tensorflow as tf
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_DIR = "kaggle-model/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
MODEL_SAVE_PATH = "models/plant_disease_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

def build_model(num_classes):
    logging.info("Building MobileNetV3Large model...")
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    
    # Data Augmentation
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    
    # Preprocessing
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def main():
    if not os.path.exists(TRAIN_DIR):
        logging.error(f"Training directory not found: {TRAIN_DIR}")
        return

    logging.info("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    logging.info(f"Found {len(class_names)} classes: {class_names}")

    # Save class names
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_names, f)
    logging.info(f"Saved class indices to {CLASS_INDICES_PATH}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_model(len(class_names))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    logging.info("Starting training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )
        
        logging.info("Training complete.")
        
        # Save model
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Model saved to {MODEL_SAVE_PATH}")

        # Optional: Unfreeze and fine-tune
        # logging.info("Fine-tuning...")
        # base_model.trainable = True
        # model.compile(...)
        # model.fit(...)
        
    except Exception as e:
        logging.error(f"Error during training: {e}")

if __name__ == "__main__":
    main()
