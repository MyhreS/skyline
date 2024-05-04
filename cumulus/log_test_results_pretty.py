import math
import plotly.graph_objects as go
import pandas as pd
import os

"""
Function for logging test results to a file
"""


def extract_classes_from_confusion_matrix(confusion_matrix: pd.DataFrame):
    classes_as_list = list(confusion_matrix.index)
    return classes_as_list


def get_class_beloning_to_label(label, class_label_map):
    for class_, labels in class_label_map.items():
        if label in labels:
            return class_

    raise ValueError(f"Label {label} not found in class_label_map")


def log_drone_results_pretty(test_results, class_label_map, classes):
    new_rows = []
    for dataset_name, results in test_results.items():
        label = dataset_name.split("test_")[1]
        confusion_matrix = results["confusion_matrix"]
        label_belongs_to_class = get_class_beloning_to_label(label, class_label_map)

        # Start with a base dictionary
        label_formated = label.replace("_", " ")
        label_formated = label.replace("-", " ")
        label_formated = label_formated[0].upper() + label_formated[1:]
        new_row = {
            "Label": label_formated,
            "True class": label_belongs_to_class,
        }

        for index, row in confusion_matrix.iterrows():
            all_columns_empty = True
            for class_ in classes:
                if row[class_] != 0:
                    all_columns_empty = False
            if not all_columns_empty:
                for class_ in classes:
                    new_row[class_] = int(row[class_])

        # Predictions from a drone perspective
        drone_predictions = 0
        other_predictions = 0
        for class_ in classes:
            if (
                "drone" in class_
                and "non-drone" not in class_
                and "non_drone" not in class_
            ):
                drone_predictions += new_row[class_]
            else:
                other_predictions += new_row[class_]
        true_class = (
            "drone"
            if "drone" in label_belongs_to_class
            and "non-drone" not in label_belongs_to_class
            and "non_drone" not in label_belongs_to_class
            else "other"
        )

        tp = fn = fp = tn = 0
        if true_class == "drone":
            tp = drone_predictions  # True positives: Correctly predicted as drone
            fn = other_predictions  # False negatives: Incorrectly predicted as other
        elif true_class == "other":
            tn = other_predictions  # True negatives: Correctly predicted as other
            fp = drone_predictions  # False positives: Incorrectly predicted as drone

        # Calculate the accuracy
        total_predictions = tp + tn + fp + fn
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        new_row["Drone TP"] = int(tp)
        new_row["Drone TN"] = int(tn)
        new_row["Drone FP"] = int(fp)
        new_row["Drone FN"] = int(fn)
        new_row["Drone FPR"] = round(fpr, 2)
        new_row["Drone Accuracy"] = math.floor(accuracy * 100) / 100.0
        new_row["Evaluation Accuracy"] = (
            math.floor(results["evaluate_accuracy"] * 100) / 100.0
        )
        new_row["Evaluation Loss"] = round(results["evaluate_loss"], 2)

        new_rows.append(new_row)

    # Make a dataframe from the new rows
    df = pd.DataFrame(new_rows)
    df.index = df["Label"]
    df.drop(columns=[("Label")], inplace=True)

    class_tuples = []
    for class_ in classes:
        class_tuples.append(("Predicted Classes", class_))

    df.columns = pd.MultiIndex.from_tuples(
        [
            ("Test Dataset Info", "True class"),
            *class_tuples,  # This will dynamically insert your class columns
            ("Binary drone metrics", "Drone TP"),
            ("Binary drone metrics", "Drone TN"),
            ("Binary drone metrics", "Drone FP"),
            ("Binary drone metrics", "Drone FN"),
            ("Binary drone metrics", "Drone FPR"),
            ("Binary drone metrics", "Drone Accuracy"),
            ("General metrics", "Evaluation Accuracy"),
            ("General metrics", "Evaluation Loss"),
        ]
    )
    # Create a summary df with the total TP, TN, FP, FN and average accuracy and FPR
    summary_df = pd.DataFrame(
        {
            ("Binary drone metrics", "Drone TP"): [
                int(df["Binary drone metrics"]["Drone TP"].sum())
            ],
            ("Binary drone metrics", "Drone TN"): [
                int(df["Binary drone metrics"]["Drone TN"].sum())
            ],
            ("Binary drone metrics", "Drone FP"): [
                int(df["Binary drone metrics"]["Drone FP"].sum())
            ],
            ("Binary drone metrics", "Drone FN"): [
                int(df["Binary drone metrics"]["Drone FN"].sum())
            ],
            ("Binary drone metrics", "Drone FPR"): [
                round(df["Binary drone metrics"]["Drone FPR"].mean(), 2)
            ],
            ("Binary drone metrics", "Drone Accuracy"): [
                math.floor(df["Binary drone metrics"]["Drone Accuracy"].mean() * 100)
                / 100.0
            ],
            ("General metrics", "Evaluation Accuracy"): [
                math.floor(df["General metrics"]["Evaluation Accuracy"].mean() * 100)
                / 100.0
            ],
            ("General metrics", "Evaluation Loss"): [
                round(df["General metrics"]["Evaluation Loss"].mean(), 2)
            ],
        },
        index=["Summary"],
    )

    final_df = pd.concat([df, summary_df])

    return final_df, df, summary_df


def log_general_results_pretty(test_results, class_label_map, classes):
    new_rows = []
    for dataset_name, results in test_results.items():
        label = dataset_name.split("test_")[1]
        confusion_matrix = results["confusion_matrix"]
        label_belongs_to_class = get_class_beloning_to_label(label, class_label_map)

        # Start with a base dictionary
        label_formated = label.replace("_", " ")
        label_formated = label_formated[0].upper() + label_formated[1:]
        new_row = {
            "Label": label_formated,
            "True class": label_belongs_to_class,
        }

        for index, row in confusion_matrix.iterrows():
            all_columns_empty = True
            for class_ in classes:
                if row[class_] != 0:
                    all_columns_empty = False
            if not all_columns_empty:
                for class_ in classes:
                    new_row[class_] = int(row[class_])
        new_rows.append(new_row)

        new_row["Evaluation Accuracy"] = round(results["evaluate_accuracy"], 2)
        new_row["Evaluation Loss"] = round(results["evaluate_loss"], 2)

    # Make a dataframe from the new rows
    df = pd.DataFrame(new_rows)
    df.index = df["Label"]
    df.drop(columns=[("Label")], inplace=True)

    class_tuples = []
    for class_ in classes:
        class_tuples.append(("Predicted Classes", class_))

    df.columns = pd.MultiIndex.from_tuples(
        [
            ("Test Dataset Info", "True class"),
            *class_tuples,
            ("General metrics", "Evaluation Accuracy"),
            ("General metrics", "Evaluation Loss"),
        ]
    )

    summary_df = pd.DataFrame(
        {
            ("General metrics", "Evaluation Accuracy"): [
                round(df["General metrics"]["Evaluation Accuracy"].mean(), 2)
            ],
            ("General metrics", "Evaluation Loss"): [
                round(df["General metrics"]["Evaluation Loss"].mean(), 2)
            ],
        },
        index=["Summary"],
    )

    final_df = pd.concat([df, summary_df])

    return final_df, df, summary_df


def sort_by_drone(df):
    df["sort"] = 0
    for index, row in df.iterrows():
        print(index)
        if "drone" in index and "non-drone" not in index and "non_drone" not in index:
            df.at[index, "sort"] = 1
        elif "Summary" in index:
            df.at[index, "sort"] = 2
        else:
            df.at[index, "sort"] = 0
    df = df.sort_values(by="sort", ascending=True)
    df.drop(columns=["sort"], inplace=True)
    return df


def log_test_results_pretty(test_results: dict, class_label_map, run_id: str):
    table_test_results_path = os.path.join("cache", run_id, "test_results_metrics.csv")
    summary_test_results_path = os.path.join(
        "cache", run_id, "test_results_summary.csv"
    )
    table_and_summary_test_results_path = os.path.join(
        "cache", run_id, "test_results_metrics_and_summary.csv"
    )

    if not os.path.exists(os.path.dirname(table_test_results_path)):
        os.makedirs(os.path.dirname(table_test_results_path))

    first_confusion_matrix = list(test_results.values())[0]["confusion_matrix"]
    classes = extract_classes_from_confusion_matrix(first_confusion_matrix)

    # Check if "drone" is in any of the classes
    if any(
        "drone" in class_ and "non-drone" not in class_ and "non_drone" not in class_
        for class_ in classes
    ):
        summary_and_metrics_df, df, df_summary = log_drone_results_pretty(
            test_results, class_label_map, classes
        )
    else:
        summary_and_metrics_df, df, df_summary = log_general_results_pretty(
            test_results, class_label_map, classes
        )
    summary_and_metrics_df = sort_by_drone(summary_and_metrics_df)
    df = sort_by_drone(df)

    df.to_csv(table_test_results_path, index=True)
    df_summary.to_csv(summary_test_results_path, index=True)
    summary_and_metrics_df.to_csv(table_and_summary_test_results_path, index=True)
    print(summary_and_metrics_df)

    save_as_plotly_plot(run_id, classes, summary_and_metrics_df)


def save_as_plotly_plot(run_id, classes, summary_and_metrics_df):
    color_map = {
        "Test Dataset Info": "#E6F5C9",
        "Predicted Classes": "lightgreen",
        "Binary drone metrics": "lightblue",
        "General metrics": "paleturquoise",
    }

    summary_and_metrics_df = summary_and_metrics_df.reset_index()
    summary_and_metrics_df.columns = summary_and_metrics_df.columns.get_level_values(1)

    # Create a mapping from the second level to the first level

    level_2_to_level_1 = {
        "": "Test Dataset Info",
        "True class": "Test Dataset Info",
        "Drone TP": "Binary drone metrics",
        "Drone TN": "Binary drone metrics",
        "Drone FP": "Binary drone metrics",
        "Drone FN": "Binary drone metrics",
        "Drone FPR": "Binary drone metrics",
        "Drone Accuracy": "Binary drone metrics",
        "Evaluation Accuracy": "General metrics",
        "Evaluation Loss": "General metrics",
    }

    for class_ in classes:
        level_2_to_level_1[class_] = "Predicted Classes"

    header_colors = [
        color_map[level_2_to_level_1[col]] for col in summary_and_metrics_df.columns
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(summary_and_metrics_df.columns),
                    fill_color=header_colors,
                    align="center",
                ),
                cells=dict(
                    values=summary_and_metrics_df.T.values.tolist(),
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        width=2600,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.write_image(os.path.join("cache", run_id, "test_results_metrics_summary.png"))
    fig.write_image(os.path.join("cache", run_id, "test_results_metrics_summary.pdf"))
