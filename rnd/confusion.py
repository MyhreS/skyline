import tensorflow as tf

# Assume y_true and y_pred are your lists of true labels and predictions, respectively
# They should be in a format that's compatible with tf.math.confusion_matrix, typically as integer labels
y_true = [2, 1, 0, 2, 2, 0, 1, 1, 0]
y_pred = [0, 1, 0, 2, 2, 0, 2, 1, 0]

# Calculate the confusion matrix
conf_matrix = tf.math.confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print(conf_matrix)