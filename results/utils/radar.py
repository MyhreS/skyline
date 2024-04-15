import pandas as pd
import plotly.express as px


# df = pd.read_csv("test_results_multiclass_perspective.csv")

# experiment_averages = df[df["Multi class perpective"].str.contains("Average")]


# melted_df = experiment_averages.melt(
#     id_vars=["Experiment"],
#     value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
#     var_name="Metric",
#     value_name="Value",
# )

# fig = px.line_polar(
#     melted_df,
#     r="Value",
#     theta="Metric",
#     color="Experiment",
#     line_close=True,
# )
# fig.update_layout(margin=dict(l=25, r=0, t=25, b=30))
# fig.write_image("figures/multiclass_test_results_radar_chart.png")


df = pd.read_csv("test_results_binary_perspective.csv")


melted_df = df.melt(
    id_vars=["Experiment"],
    value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
    var_name="Metric",
    value_name="Value",
)

fig = px.line_polar(
    melted_df,
    r="Value",
    theta="Metric",
    color="Experiment",
    line_close=True,
)
fig.update_layout(margin=dict(l=25, r=0, t=25, b=30))
fig.write_image("figures/binaryclass_test_results_radar_chart.png")
