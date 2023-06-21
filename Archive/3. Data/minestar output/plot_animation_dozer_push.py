import numpy as np
import plotly.graph_objects as go
import pandas as pd

# test_cycle = pd.read_csv("dozer_fms_stats_VER_2023-02-08_Day.csv")
# test_cycle = test_cycle.iloc[6811:6877]  # [333:396] #[21:80] #[2206:2246] #[6811:6877]

test_cycle = pd.read_csv("dozer_fms_stats_VER_2023-02-09_Night.csv")
test_cycle = test_cycle.iloc[32:88]  # [333:396] #[21:80] #[2206:2246] #[6811:6877]
test_cycle = test_cycle.reset_index(drop=True)

# Create figure
fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(color="red", size=10))])

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[min(test_cycle.easting_m), max(test_cycle.easting_m)], autorange=False),
        yaxis=dict(range=[min(test_cycle.northing_m), max(test_cycle.northing_m)], autorange=False),
        zaxis=dict(range=[min(test_cycle.elevation_m), max(test_cycle.elevation_m)], autorange=False),
    )
),


frames = [
    go.Frame(
        data=[
            go.Scatter3d(
                x=test_cycle.easting_m[: k + 1],
                y=test_cycle.northing_m[: k + 1],
                z=test_cycle.elevation_m[: k + 1],
                marker=dict(color=test_cycle.loc[: k + 1, "speed_smoothed_km/hr"]),
            )
        ],
        traces=[0],
        name=f"frame{k}",
    )
    for k in range(len(test_cycle.easting_m) - 1)
]
fig.update(frames=frames)


fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(redraw=True, fromcurrent=True, mode="immediate"))],
                )
            ],
        )
    ]
)


# fig.show()


###### Test 3d cone plots #####
test_cycle = pd.read_csv("dozer_fms_stats_VER_2023-02-09_Night.csv")
test_cycle = test_cycle.iloc[4942:4999]

test_cycle["u_m"] = np.sin(np.radians(90) - np.arctan(test_cycle["grade_%"] / 100)) * np.cos(
    np.radians(test_cycle["track_smoothed_degrees"])  # + 90
)
test_cycle["v_m"] = np.sin(np.radians(90) - np.arctan(test_cycle["grade_%"] / 100)) * np.sin(
    np.radians(test_cycle["track_smoothed_degrees"])  # + 90
)
test_cycle["w_m"] = np.cos(np.radians(90) - np.arctan(test_cycle["grade_%"] / 100))

# Create figure
fig = go.Figure(data=[go.Cone(x=[], y=[], z=[], u=[], v=[], w=[])])

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[min(test_cycle.easting_m), max(test_cycle.easting_m)], autorange=False),
        yaxis=dict(range=[min(test_cycle.northing_m), max(test_cycle.northing_m)], autorange=False),
        zaxis=dict(range=[min(test_cycle.elevation_m), max(test_cycle.elevation_m)], autorange=False),
    )
),


frames = [
    go.Frame(
        data=[
            go.Cone(
                x=test_cycle.easting_m[: k + 1],
                y=test_cycle.northing_m[: k + 1],
                z=test_cycle.elevation_m[: k + 1],
                u=test_cycle.u_m[: k + 1],
                v=test_cycle.v_m[: k + 1],
                w=test_cycle.w_m[: k + 1],
            )
        ],
        traces=[0],
        name=f"frame{k}",
    )
    for k in range(len(test_cycle.easting_m) - 1)
]
fig.update(frames=frames)


fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(redraw=True, fromcurrent=True, mode="immediate"))],
                )
            ],
        )
    ]
)
# fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="data"))

# fig.show()  # fixed

# As of 2020 it was not possible to allocate colour to cone plot

# Next step may be to try and do this in matplotlib


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ax = plt.figure().add_subplot(projection="3d")
# norm = matplotlib.colors.Normalize()
# norm.autoscale(test_cycle["speed_smoothed_km/hr"])
# cm = matplotlib.cm.copper
# sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
# sm.set_array([])
# ax.quiver(
#     test_cycle.easting_m,
#     test_cycle.northing_m,
#     test_cycle.elevation_m,
#     test_cycle.u_m,
#     test_cycle.v_m,
#     test_cycle.w_m,
#     length=2,
#     normalize=True,
#     color=cm(norm(test_cycle["speed_smoothed_km/hr"])),
# )

# plt.colorbar(sm)
# plt.show()

##### Test matplotlib 3D scatter animation #####
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

cmap = matplotlib.cm.jet
norm = matplotlib.colors.Normalize(
    vmin=int(test_cycle["speed_smoothed_km/hr"].min()) - 1, vmax=int(test_cycle["speed_smoothed_km/hr"].max()) + 1
)


frame_t = []
for i, item in enumerate(test_cycle.delta_time_prev_sec):
    frame_t.extend([i] * int(test_cycle.delta_time_prev_sec.iloc[i]))


def update_graph(num):
    data = test_cycle.iloc[:num]
    graph._offsets3d = (data.easting_m, data.northing_m, data.elevation_m)
    graph.set_color(cmap(norm(data["speed_smoothed_km/hr"][:num])))
    graph.set_alpha(0.5)
    title.set_text("Example Push Cycle, position={}".format(num))


fig = plt.figure(figsize=(12.0, 7.0))
ax = fig.add_subplot(111, projection="3d")
title = ax.set_title("Example Push Cycle")

data = test_cycle.iloc[:1]
graph = ax.scatter(
    xs=data.easting_m,
    ys=data.northing_m,
    zs=data.elevation_m,
    # c=data["speed_smoothed_km/hr"],
    # vmin=int(test_cycle["speed_smoothed_km/hr"].min()) - 1,
    # vmax=int(test_cycle["speed_smoothed_km/hr"].max()) + 1,
    # cmap=cmap,
)
ax.azim = -35
ax.elev = 24
ax.set_aspect("equal")
dummy = ax.scatter(
    xs=test_cycle.easting_m * 1000,
    ys=test_cycle.northing_m * 1000,
    zs=test_cycle.elevation_m,
    c=test_cycle["speed_smoothed_km/hr"],
    vmin=int(test_cycle["speed_smoothed_km/hr"].min()) - 1,
    vmax=int(test_cycle["speed_smoothed_km/hr"].max()) + 1,
    cmap=cmap,
)
ani = matplotlib.animation.FuncAnimation(fig, update_graph, frame_t, interval=100, blit=False)
cbar = plt.colorbar(dummy)
cbar.set_label("Speed (km/hr)")
plt.xlim([test_cycle.easting_m.min() - 10, test_cycle.easting_m.max() + 10])
plt.ylim([test_cycle.northing_m.min() - 10, test_cycle.northing_m.max() + 10])
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.show()

ani.save("test_gif.gif", writer="MovieWriter", fps=30, dpi=300)
### Sucess!!!!!!!!
