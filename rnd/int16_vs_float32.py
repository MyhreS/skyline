import matplotlib.pyplot as plt
import numpy as np


# Create a random float32 array (one second with sampel rate of 44100)
float32_array = np.random.rand(44100).astype(np.float32)

# Convert to int16
int16_array = (float32_array * 32767).astype(np.int16)

# Plot the two arrays
plt.figure(figsize=(10, 5))
plt.plot(float32_array, label="float32")
plt.plot(int16_array, label="int16")
plt.legend()
plt.show()

