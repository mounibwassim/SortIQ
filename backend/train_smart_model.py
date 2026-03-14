import os
import glob
import json
import random
import shutil
import numpy as np  # pyre-ignore
import tensorflow as tf  # pyre-ignore
from tensorflow import keras  # pyre-ignore
from tensorflow.keras import layers, models  # pyre-ignore
from tensorflow.keras.applications import MobileNetV2  # pyre-ignore
import kagglehub  # pyre-ignore
from sklearn.model_selection import train_test_split  # pyre-ignore

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Download and subset Kaggle dataset
def get_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mostafaabla/garbage-classification")
    root_dir = os.path.join(path, "garbage_classification")
    
    # Target classes matching UI requirements
    target_classes = ["plastic", "paper", "metal", "glass"]
    class_mapping = {
        "plastic": "Plastic",
        "paper": "Paper",
        "metal": "Metal",
        "glass": "Glass"
    }
    
    filepaths = []
    labels = []
    
    print("\nExtracting subsets for local training...")
    # Map index 0->Glass, 1->Metal, 2->Paper, 3->Plastic
    label_map = {"Glass": 0, "Metal": 1, "Paper": 2, "Plastic": 3}
    
    for folder in os.listdir(root_dir):
        if folder.lower() in target_classes:
            class_name = class_mapping[folder.lower()]
            label_idx = label_map[class_name]
            folder_path = os.path.join(root_dir, folder)
            
            # Get all images
            imgs = glob.glob(os.path.join(folder_path, "*.jpg"))
            
            # Subsample to 200 images per class for feasible local CPU training time
            sampled_imgs = random.sample(imgs, min(200, len(imgs)))
            
            filepaths.extend(sampled_imgs)
            labels.extend([label_idx] * len(sampled_imgs))
            
            print(f"Loaded {len(sampled_imgs)} images for {class_name}")

    return train_test_split(filepaths, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

# 2. Preprocess and Augment Data
def process_path(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def build_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(1000, seed=RANDOM_SEED)
        
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    if training:
        # Data Augmentation identical to notebook
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ])
        # Need to wrap label in lambda correctly
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# 3. Build Transfer Learning Model
def build_model(num_classes=4):
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    train_paths, val_paths, train_labels, val_labels = get_dataset()
    
    # Convert labels to categorical
    train_labels = keras.utils.to_categorical(train_labels, num_classes=4)
    val_labels = keras.utils.to_categorical(val_labels, num_classes=4)
    
    print("Building TensorFlow Datasets...")
    train_ds = build_dataset(train_paths, train_labels, training=True)
    val_ds = build_dataset(val_paths, val_labels, training=False)
    
    print("Building MobileNetV2...")
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training the authentic model (1 epoch for time constraints)...")
    # For a real pipeline, 10+ epochs are used, but we stick to 1 to bound CPU time while producing authentic weights
    model.fit(train_ds, validation_data=val_ds, epochs=1)
    
    # Save Model
    model_dir = os.path.join(BASE_DIR, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'sortiq_model.h5')
    model.save(model_path)
    print(f"\n✅ Authentic SortIQ MobileNetV2 saved to {model_path}")
    
    # Output class JSON
    classes_path = os.path.join(model_dir, 'classes.json')
    with open(classes_path, 'w') as f:
        json.dump({"0": "Glass", "1": "Metal", "2": "Paper", "3": "Plastic"}, f, indent=2)

if __name__ == "__main__":
    main()
