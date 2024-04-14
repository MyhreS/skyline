import json
import plotly.express as px
import pandas as pd

with open("../Run_1/train_history.json", "r") as f:
    run_1_history = json.load(f)

with open("../Run_2/train_history.json", "r") as f:
    run_2_history = json.load(f)

with open("../Run_3/train_history.json", "r") as f:
    run_3_history = json.load(f)

with open("../Run_4/train_history.json", "r") as f:
    run_4_history = json.load(f)

with open("../Run_5/train_history.json", "r") as f:
    run_5_history = json.load(f)

with open("../Run_6/train_history.json", "r") as f:
    run_6_history = json.load(f)

with open("../Run_7/train_history.json", "r") as f:
    run_7_history = json.load(f)


# COnvert the dictionaries to a single DataFrame
experiments = [
    run_1_history,
    run_2_history,
    run_3_history,
    run_4_history,
    run_5_history,
    run_6_history,
    run_7_history,
]


def smooth_curve(points, factor=0.5):
    smoothed_points = points.copy()
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points


df = pd.DataFrame()
for i, run in enumerate(experiments):
    experiment = {
        "experiment": i + 1,
        "accuracy": smooth_curve(run["accuracy"]),
        "val_accuracy": smooth_curve(run["val_accuracy"]),
        "loss": smooth_curve(run["loss"]),
        "val_loss": smooth_curve(run["val_loss"]),
    }
    df = pd.concat([df, pd.DataFrame(experiment)])


df_long = pd.melt(
    df.reset_index(),
    id_vars=["index", "experiment"],
    value_vars=["accuracy", "val_accuracy"],
    value_name="Value",
)

df_long.columns = ["Epoch", "Experiment", "Metric", "Value"]

fig = px.line(
    df_long,
    x="Epoch",
    y="Value",
    color="Experiment",
    facet_row="Metric",
    labels={"Epoch": "Epoch", "Value": "Metric Value"},
)
for annotation in fig.layout.annotations:
    annotation.text = ""
fig.layout.yaxis.title.text = "Validation Accuracy"
fig.layout.yaxis2.title.text = "Accuracy"


# set margins to 0
fig.update_layout(margin=dict(l=25, r=0, t=30, b=30))

# fig.show()
fig.write_image("figures/accuracy_history.png")


df_long = pd.melt(
    df.reset_index(),
    id_vars=["index", "experiment"],
    value_vars=["loss", "val_loss"],
)

df_long.columns = ["Epoch", "Experiment", "Metric", "Value"]

fig = px.line(
    df_long,
    x="Epoch",
    y="Value",
    color="Experiment",
    facet_row="Metric",
    labels={"Epoch": "Epoch", "Value": "Metric Value"},
)
for annotation in fig.layout.annotations:
    annotation.text = ""
fig.layout.yaxis.title.text = "Validation Loss"
fig.layout.yaxis2.title.text = "Loss"

fig.update_layout(margin=dict(l=25, r=0, t=30, b=30))
fig.write_image("figures/loss_history.png")

# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(run_1_history)


# # Define a function to apply moving average smoothing
# def smooth_curve(points, factor=0.5):
#     smoothed_points = points.copy()
#     for i in range(1, len(points)):
#         smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
#     return smoothed_points


# # Apply smoothing to the accuracy and validation accuracy
# df["accuracy"] = smooth_curve(df["accuracy"].tolist())
# df["val_accuracy"] = smooth_curve(df["val_accuracy"].tolist())
# df["loss"] = smooth_curve(df["loss"].tolist())
# df["val_loss"] = smooth_curve(df["val_loss"].tolist())

# """
# Plot accuracy
# """

# df_long = pd.melt(
#     df.reset_index(), id_vars=["index"], value_vars=["accuracy", "val_accuracy"]
# )
# df_long.columns = ["Epoch", "Metric", "Value"]

# fig = px.line(
#     df_long,
#     x="Epoch",
#     y="Value",
#     color="Metric",
#     labels={"Epoch": "Epoch", "Value": "Accuracy"},
#     title="Training and Validation Accuracy over Epochs",
# )

# fig.update_layout(width=800, height=500)
# fig.write_image(f"{RUN}/accuracy_history.png")

# """
# Plot loss
# """

# df_long = pd.melt(df.reset_index(), id_vars=["index"], value_vars=["loss", "val_loss"])
# df_long.columns = ["Epoch", "Metric", "Value"]

# fig = px.line(
#     df_long,
#     x="Epoch",
#     y="Value",
#     color="Metric",
#     labels={"Epoch": "Epoch", "Value": "Loss"},
#     title="Training and Validation Loss over Epochs",
# )
# fig.update_layout(width=800, height=500)
# fig.write_image(f"{RUN}/loss_history.png")
