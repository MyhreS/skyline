import pandas as pd
import plotly.express as px

"""
Read the csv
"""

df = pd.read_csv("multiclass_metrics.csv")
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    df[col] = df[col].str.replace(",", ".").astype(float)


experiment_averages = (
    df.groupby("Experiment")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    .mean()
    .reset_index()
)


melted_df = experiment_averages.melt(id_vars=["Experiment"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], 
                                     var_name="Metric", value_name="Value")

# Now, create the radar chart
fig = px.line_polar(
    melted_df,
    r="Value",
    theta="Metric",
    color="Experiment",
    line_close=True,
)
fig.update_layout(margin=dict(l=25, r=0, t=30, b=30))
fig.write_image("multiclass_metrics_radar_chart.png")
