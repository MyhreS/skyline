import pandas as pd
import plotly.express as px

"""
Bar graph accuracy
"""

# # Data preparation
# results = {
#     "Experiment 1": {"Drone accuracy": 92, "Accuracy": 92},
#     "Experiment 2": {"Drone accuracy": 90, "Accuracy": 77},
#     "Experiment 3": {"Drone accuracy": 91, "Accuracy": 86},
#     "Experiment 4": {"Drone accuracy": 91, "Accuracy": 81},
#     "Experiment 5": {"Drone accuracy": 89, "Accuracy": 85},
#     "Experiment 6": {"Drone accuracy": 90, "Accuracy": 78},
#     "Experiment 7": {"Drone accuracy": 85, "Accuracy": 66},
# }

# # Transforming the data into a format suitable for Plotly Express
# data = {"Experiments": [], "Metric": [], "Accuracy (%)": []}

# for experiment, metrics in results.items():
#     for metric, value in metrics.items():
#         data["Experiments"].append(experiment)
#         data["Metric"].append(metric)
#         data["Accuracy (%)"].append(value)

# df = pd.DataFrame(data)

# # Creating the bar chart
# fig = px.bar(
#     df,
#     x="Experiments",
#     y="Accuracy (%)",
#     color="Metric",
#     barmode="group",
#     color_discrete_sequence=px.colors.qualitative.Pastel,
#     height=400,
#     text_auto=True,
# )
# fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(range=[60, 100]))
# # Show the plot
# fig.show()
# fig.write_image("accuracy_bar_chart.png")

# Data preparation


# """
# Multiclass perspective
# """
# results = {
#     "Experiment 1": {"Accuracy": 92, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
#     "Experiment 2": {"Accuracy": 77, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
#     "Experiment 3": {"Accuracy": 86, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
#     "Experiment 4": {"Accuracy": 81, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
#     "Experiment 5": {"Accuracy": 85, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
#     "Experiment 6": {"Accuracy": 78, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
#     "Experiment 7": {"Accuracy": 66, "FPs": 0, "FNs": 0, "TPs": 0, "TNs": 0},
# }

# # Transforming the data into a format suitable for Plotly Express
# data = {"Experiments": [], "Metric": [], "Accuracy (%)": []}

# for experiment, metrics in results.items():
#     for metric, value in metrics.items():
#         data["Experiments"].append(experiment)
#         data["Metric"].append(metric)
#         data["Accuracy (%)"].append(value)

# df = pd.DataFrame(data)

# # Creating the bar chart
# fig = px.bar(
#     df,
#     x="Experiments",
#     y="Accuracy (%)",
#     color="Metric",
#     barmode="group",
#     color_discrete_sequence=px.colors.qualitative.Pastel,
#     height=400,
#     text_auto=True,
# )
# fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(range=[60, 100]))
# # Show the plot
# fig.show()
# fig.write_image("accuracy_bar_chart.png")


results = {
    "Experiment 1": {"FPs": 787, "FNs": 602, "TPs": 8504, "TNs": 9495},
    "Experiment 2": {"FPs": 811, "FNs": 825, "TPs": 8281, "TNs": 9471},
    "Experiment 3": {"FPs": 762, "FNs": 707, "TPs": 8399, "TNs": 9520},
    "Experiment 4": {"FPs": 866, "FNs": 755, "TPs": 8351, "TNs": 9416},
    "Experiment 5": {"FPs": 1012, "FNs": 831, "TPs": 8275, "TNs": 9270},
    "Experiment 6": {"FPs": 689, "FNs": 836, "TPs": 8270, "TNs": 7039},
    "Experiment 7": {"FPs": 1744, "FNs": 431, "TPs": 8675, "TNs": 5984},
}

# Transforming the data into a format suitable for Plotly Express
data = {"Experiments": [], "Metric": [], "Amount": []}

for experiment, metrics in results.items():
    for metric, value in metrics.items():
        data["Experiments"].append(experiment)
        data["Metric"].append(metric)
        data["Amount"].append(value)

df = pd.DataFrame(data)

# Creating the bar chart
fig = px.bar(
    df,
    x="Experiments",
    y="Amount",
    color="Metric",
    barmode="group",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    # height=800,
    width=1800,
    text_auto=True,
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
# Show the plot
fig.show()
fig.write_image("Binary_drone_FP_FN_TP_TN.png")
