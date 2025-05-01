import tensorflow as tf

# 1. Load your saved model (no compile needed)
model = tf.keras.models.load_model('models/throughput_model.keras', compile=False)

# 2. Save just the weights with the correct extension
model.save_weights('models/throughput_weights.weights.h5')
print("✅ Weights written to models/throughput_weights.weights.h5")
