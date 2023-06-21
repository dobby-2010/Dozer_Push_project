import pandas as pd
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statistics as stat

# For all dozers plot out to see if the conditions for relocation are the same or if there is some variance


# Pseudo code to determine the relocation part if any for a dozer
# Meant for after general classification. Ca test this on highly classified data and non-classified data - VER
def _identify_relocations(positions, distance_threshold: float = 600, angle_threshold: float = 30):

    for dozer in positions.Equipment_Code.unique():
        data_dozer = positions[positions["Equipment_Code"] == dozer]
        dz_agg = data_dozer.groupby(["Straight_Index_Start", "Straight_Index_Stop", "Straight_Type"]).agg(
            {"Straight_Distance (m)": "mean"}
        )
        dz_agg = dz_agg.reset_index()
        dozer_avg_str_dist = stat.mean(dz_agg["Straight_Distance (m)"])

        degree_list = []
        for i in range(len(dz_agg)):
            reg = LinearRegression().fit(
                np.array(
                    data_dozer.loc[
                        dz_agg.iloc[i].Straight_Index_Start.astype(int) : dz_agg.iloc[i].Straight_Index_Stop.astype(int)
                    ]["Easting (m)"]
                ).reshape((-1, 1)),
                np.array(
                    data_dozer.loc[
                        dz_agg.iloc[i].Straight_Index_Start.astype(int) : dz_agg.iloc[i].Straight_Index_Stop.astype(int)
                    ]["Northing (m)"]
                ),
            )
            degree_list.append(np.degrees(np.arctan(reg.coef_[0])))

        dz_agg["Straight Direction (Deg)"] = degree_list
        dozer_avg_dir = stat.mean(degree_list)

        potential_relocations = []
        for i in range(len(dz_agg)):
            if (
                (dz_agg.iloc[i]["Straight_Distance (m)"] / dozer_avg_str_dist) * 100 > distance_threshold
                and (dz_agg.iloc[i]["Straight_Distance (m)"] / dozer_avg_dir) * 100 > angle_threshold
                and dz_agg.iloc[i]["Straight_Type"] == "unclassified"
            ):
                potential_relocations.append(i)
            else:
                pass

        for i in potential_relocations:
            if dz_agg.iloc[i]["Straight_Index_Start"] == dz_agg.iloc[i]["Straight_Index_Stop"] - 1:
                positions.loc[dz_agg.iloc[i]["Straight_Index_Start"], "Straight_Type"] = "relocate"
            else:
                positions.loc[
                    dz_agg.iloc[i]["Straight_Index_Start"]
                    .astype(int) : dz_agg.iloc[i]["Straight_Index_Stop"]
                    .astype(int)
                    - 1,
                    "Straight_Type",
                ] = "relocate"

    return positions
