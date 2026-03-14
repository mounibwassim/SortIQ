import os
import json
import tensorflow as tf  # pyre-ignore
from tensorflow import keras  # pyre-ignore
from tensorflow.keras import layers  # pyre-ignore

def create_mock():
    # 1. Define model architecture identical to our notebook target
    input_shape = (224, 224, 3)
    num_classes = 4
    
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 2. Ensure model directory exists
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 3. Save the model
    model_path = os.path.join(model_dir, 'sortiq_model.h5')
    model.save(model_path)
    print(f"Mock model saved to {model_path}")
    
    # 4. Save the classes JSON
    classes_path = os.path.join(model_dir, 'classes.json')
    class_mapping = {
        "0": "Glass",
        "1": "Metal",
        "2": "Paper",
        "3": "Plastic"
    }
    with open(classes_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Classes mapping saved to {classes_path}")

if __name__ == "__main__":
    create_mock()
