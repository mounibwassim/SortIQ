try:
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    import tensorflow as tf
    print("TensorFlow Version:", tf.__version__)
    from main import app
    print("App imported successfully")
except Exception:
    import traceback
    traceback.print_exc()
