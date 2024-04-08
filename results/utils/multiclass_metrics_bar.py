import pandas as pd
import plotly.express as px

"""
Read the csv
"""

df = pd.read_csv("multiclass_metrics.csv")
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    df[col] = df[col].str.replace(",", ".").astype(float)

"""
Chart for the multiclass perspective
"""

experiment_df = df[df["Experiment"] == "Experiment 1"]
fig = px.bar(
    experiment_df,
    y="Multi class perpective",
    x=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text_auto=True,
)
fig.update_yaxes(title="Classes")
fig.update_xaxes(title="Performance Metrics")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("multiclass_metrics_bar_chart_experiment_1.png")


experiment_df = df[df["Experiment"] == "Experiment 2"]
fig = px.bar(
    experiment_df,
    y="Multi class perpective",
    x=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text_auto=True,
)

fig.update_yaxes(title="Classes")
fig.update_xaxes(title="Performance Metrics")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("multiclass_metrics_bar_chart_experiment_2.png")


experiment_df = df[df["Experiment"] == "Experiment 3"]
fig = px.bar(
    experiment_df,
    y="Multi class perpective",
    x=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text_auto=True,
)

fig.update_yaxes(title="Classes")
fig.update_xaxes(title="Performance Metrics")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("multiclass_metrics_bar_chart_experiment_3.png")


experiment_df = df[df["Experiment"] == "Experiment 4"]
fig = px.bar(
    experiment_df,
    y="Multi class perpective",
    x=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text_auto=True,
)

fig.update_yaxes(title="Classes")
fig.update_xaxes(title="Performance Metrics")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("multiclass_metrics_bar_chart_experiment_4.png")

experiment_df = df[df["Experiment"] == "Experiment 5"]
fig = px.bar(
    experiment_df,
    y="Multi class perpective",
    x=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text_auto=True,
)

fig.update_yaxes(title="Classes")
fig.update_xaxes(title="Performance Metrics")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("multiclass_metrics_bar_chart_experiment_5.png")


