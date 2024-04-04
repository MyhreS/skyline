import json
import plotly.express as px
import pandas as pd

RUN = "Run_1"

# Read train_history.json
with open(f"{RUN}/train_history.json", "r") as f:
    train_history = json.load(f)

# Convert the dictionary to a DataFrame
df = pd.DataFrame(train_history)


# Define a function to apply moving average smoothing
def smooth_curve(points, factor=0.5):
    smoothed_points = points.copy()
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points


# Apply smoothing to the accuracy and validation accuracy
df["accuracy"] = smooth_curve(df["accuracy"].tolist())
df["val_accuracy"] = smooth_curve(df["val_accuracy"].tolist())
df["loss"] = smooth_curve(df["loss"].tolist())
df["val_loss"] = smooth_curve(df["val_loss"].tolist())

"""
Plot accuracy
"""

df_long = pd.melt(
    df.reset_index(), id_vars=["index"], value_vars=["accuracy", "val_accuracy"]
)
df_long.columns = ["Epoch", "Metric", "Value"]

fig = px.line(
    df_long,
    x="Epoch",
    y="Value",
    color="Metric",
    labels={"Epoch": "Epoch", "Value": "Accuracy"},
    title="Training and Validation Accuracy over Epochs",
)

fig.update_layout(width=800, height=500)
fig.write_image(f"{RUN}/accuracy_history.png")

"""
Plot loss
"""

df_long = pd.melt(df.reset_index(), id_vars=["index"], value_vars=["loss", "val_loss"])
df_long.columns = ["Epoch", "Metric", "Value"]

fig = px.line(
    df_long,
    x="Epoch",
    y="Value",
    color="Metric",
    labels={"Epoch": "Epoch", "Value": "Loss"},
    title="Training and Validation Loss over Epochs",
)
fig.update_layout(width=800, height=500)
fig.write_image(f"{RUN}/loss_history.png")
