import pandas as pd
import plotly.express as px


data = {
    "Electric quad drone": 257782,
    "Racing drone": 13935,
    "Electric fixedwing drone": 13326,
    "Petrol fixedwing drone": 12606,
    "Misc": 178996,
    "Animal": 5611,
    "Speech": 152534,
    "Nature": 7978,
    "Urban": 146863,
}

# Total: 789,631

# Convert the data to a DataFrame
df = pd.DataFrame(list(data.items()), columns=["Category", "Duration (sec)"])

# Create the pie chart
fig = px.pie(
    df,
    values="Duration (sec)",
    names="Category",
    hover_data=["Duration (sec)"],
    labels={"Duration (sec)": "Duration (sec)"},
    color_discrete_sequence=px.colors.qualitative.Pastel,
    hole=0.3,
)

# Update the layout to show the count (duration) next to the labels
fig.update_traces(
    textinfo="percent+label", textposition="outside", pull=[0.05] * df.shape[0]
)
fig.update_layout(
    uniformtext_minsize=12,
    uniformtext_mode="hide",
    margin=dict(t=0, b=0, l=0, r=0),
    showlegend=False,
)

# Show the pie chart
# fig.show()
fig.write_image("data_duration_pie_chart.png")
