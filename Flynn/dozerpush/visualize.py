import pandas as pd
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
from pyodbc import Connection
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


# data = pd.read_csv("positions_stats.csv")
# data_ver = data[data["site"] == "VER"]
# data_son = data[data["site"] == "SON"]

# # Get % of unclassified/push/return straight type
# unclassified_ver = (
#     data_ver["straight_type"].value_counts()["unclassified"] / data_ver["straight_type"].value_counts().sum()
# ) * 100
# push_ver = (data_ver["straight_type"].value_counts()["push"] / data_ver["straight_type"].value_counts().sum()) * 100
# return_ver = (data_ver["straight_type"].value_counts()["return"] / data_ver["straight_type"].value_counts().sum()) * 100

# unclassified_son = (
#     data_son["straight_type"].value_counts()["unclassified"] / data_son["straight_type"].value_counts().sum()
# ) * 100
# push_son = (data_son["straight_type"].value_counts()["push"] / data_son["straight_type"].value_counts().sum()) * 100
# return_son = (data_son["straight_type"].value_counts()["return"] / data_son["straight_type"].value_counts().sum()) * 100

# # VER
# fig = go.Figure()
# fig = px.scatter(x=data_ver["easting"], y=data_ver["northing"], color=data_ver["straight_type"])
# fig.update_layout(
#     title_text="Order of Equipment Test VER then SON - VER: Unclassified is %s, Push is %s, and Return is %s"
#     % (unclassified_ver, push_ver, return_ver),
#     xaxis_title="Easting (m)",
#     yaxis_title="Northing (m)",
# )
# fig.update_layout(template="plotly_white")
# fig.update_layout(font=dict(size=18))
# fig.show()

# # SON
# fig = go.Figure()
# fig = px.scatter(x=data_son["easting"], y=data_son["northing"], color=data_son["straight_type"])
# fig.update_layout(
#     title_text="Order of Equipment Test VER then SON: SON: Unclassified is %s, Push is %s, and Return is %s"
#     % (unclassified_son, push_son, return_son),
#     xaxis_title="Easting (m)",
#     yaxis_title="Northing (m)",
# )
# fig.update_layout(template="plotly_white")
# fig.update_layout(font=dict(size=18))
# fig.show()

# # For reverse:
# # VER
# fig = go.Figure()
# fig = px.scatter(x=data_ver["easting"], y=data_ver["northing"], color=data_ver["straight_type"])
# fig.update_layout(
#     title_text="Order of Equipment Test SON then VER - VER: Unclassified is %s, Push is %s, and Return is %s"
#     % (unclassified_ver, push_ver, return_ver),
#     xaxis_title="Easting (m)",
#     yaxis_title="Northing (m)",
# )
# fig.update_layout(template="plotly_white")
# fig.update_layout(font=dict(size=18))
# fig.show()

# # SON
# fig = go.Figure()
# fig = px.scatter(x=data_son["easting"], y=data_son["northing"], color=data_son["straight_type"])
# fig.update_layout(
#     title_text="Order of Equipment Test SON then VER - SON: Unclassified is %s, Push is %s, and Return is %s"
#     % (unclassified_son, push_son, return_son),
#     xaxis_title="Easting (m)",
#     yaxis_title="Northing (m)",
# )
# fig.update_layout(template="plotly_white")
# fig.update_layout(font=dict(size=18))
# fig.show()

###############################################################################

# James plotting


###############################################################################
# PLOTTING FUNCTIONS
###############################################################################
dozer_state_plot_params = {
    "unclassified": {"colour": "C0", "zorder": 2.0},
    "return": {"colour": "C1", "zorder": 3.0},
    "push": {"colour": "C3", "zorder": 4.0},
}
dozer_map_padding_ratio = 0.05
dims_subplots = [10.0, 200.0, 10.0, 270.0]
dims_figure_size = [210.0, 297.0]
dims_axes_title = [
    [10.0, 200.0, 279.0, 287.0],
    [10.0, 200.0, 271.0, 279.0],
]


def add_title_string(
    fig,
    title_string,
    title_position,
    font_size,
):

    # Add title string to figure
    ax = fig.add_axes(
        [
            title_position[0] / dims_figure_size[0],
            title_position[2] / dims_figure_size[1],
            (title_position[1] - title_position[0]) / dims_figure_size[0],
            (title_position[3] - title_position[2]) / dims_figure_size[1],
        ]
    )
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        title_string,
        ha="center",
        va="center",
        fontfamily="Arial",
        fontsize=font_size,
    )


# ------------------------------------------------------------------------------
def compile_cycle_type_stat_string(
    positions,
):
    return (
        "Push: %0.1fkm, %0.1fhrs   "
        % (
            positions.loc[positions.straight_type.eq("push")].run_point.sum() / 1000.0,
            positions.loc[positions.straight_type.eq("push")].duration_point.sum() / 3600.0,
        )
        + "Return: %0.1fkm, %0.1fhrs   "
        % (
            positions.loc[positions.straight_type.eq("return")].run_point.sum() / 1000.0,
            positions.loc[positions.straight_type.eq("return")].duration_point.sum() / 3600.0,
        )
        + "Unclassified: %0.1fkm, %0.1fhrs"
        % (
            positions.loc[positions.straight_type.eq("unclassified")].run_point.sum() / 1000.0,
            positions.loc[positions.straight_type.eq("unclassified")].duration_point.sum() / 3600.0,
        ),
    )[0]


# ------------------------------------------------------------------------------
def plot_map_to_axis(
    ax,
    positions,
    background_image,  # Dictionary of PIL image (im) and extents (extents)
    status_colours={
        "unclassified": {"c": "C0", "zorder": 2.0},
        "return": {"c": "C1", "zorder": 3.0},
        "push": {"c": "C3", "zorder": 4.0},
    },
):

    # Add background image
    ax.imshow(
        background_image["im"],
        extent=background_image["extents"],
        zorder=1.0,
        cmap="gray",
    )

    # Plot each cycle type
    legend = []
    for status in ["unclassified", "return", "push"]:

        # Plot points
        ax.scatter(
            positions.loc[positions.straight_type.eq(status)].easting,
            positions.loc[positions.straight_type.eq(status)].northing,
            **status_colours[status],
        )
        legend.append(status)

    # Adjust extents
    data_extents = [
        positions.easting.min() - dozer_map_padding_ratio * (positions.easting.max() - positions.easting.min()),
        positions.easting.max() + dozer_map_padding_ratio * (positions.easting.max() - positions.easting.min()),
        positions.northing.min() - dozer_map_padding_ratio * (positions.northing.max() - positions.northing.min()),
        positions.northing.max() + dozer_map_padding_ratio * (positions.northing.max() - positions.northing.min()),
    ]
    ax.set_xlim(data_extents[0:2])
    ax.set_ylim(data_extents[2:])

    # Add labels
    ax.set_xlabel("Eastings (m)")
    ax.set_ylabel("Northings (m)")
    ax.legend(legend)


# ------------------------------------------------------------------------------
def plot_overview_map(
    site,
    shift_description,
    positions,
    background_image,  # Dictionary of PIL image (im) and extents (extents)
):

    # Initialise figure
    positions = positions.copy()
    fig = plt.figure(
        figsize=(dims_figure_size[0] / 25.4, dims_figure_size[1] / 25.4),
    )
    fig.clf()

    # Plot data
    plot_map_to_axis(fig.gca(), positions, background_image)

    # Tighten plot
    plt.tight_layout(
        pad=1.0,
        rect=[
            dims_subplots[0] / dims_figure_size[0],
            dims_subplots[2] / dims_figure_size[1],
            dims_subplots[1] / dims_figure_size[0],
            dims_subplots[3] / dims_figure_size[1],
        ],
    )

    # Add title
    title_string = "Dozer Push %s %s: Overview Map" % (
        site,
        shift_description,
    )
    add_title_string(fig, title_string, dims_axes_title[0], 16.0)
    title_string = compile_cycle_type_stat_string(positions)
    add_title_string(fig, title_string, dims_axes_title[1], 13.0)

    # Return figure
    return fig


# ------------------------------------------------------------------------------
def plot_times_distances_on_axes_OLD(
    ax,
    straight_cycles,
):

    # Put something in place
    ax2 = ax.twinx()
    ax.set_title("Cycle Distances and Times")
    ax.set_xticks([0.0, 1.0, 2.5, 3.5])
    ax.set_xticklabels(
        [
            "Distance\nPush",
            "Distance\nReturn",
            "Time\nPush",
            "Time\nReturn",
        ]
    )
    ax.set_ylabel("Distance (m)")
    ax2.set_ylabel("Time (s)")

    # Any positions?
    if straight_cycles.shape[0] == 0:
        return

    # Process times and distances
    for i, plot_info in enumerate(
        [
            ["push", "distance"],
            ["return", "distance"],
            ["push", "time"],
            ["return", "time"],
        ]
    ):

        # Distance or time plot?
        if i < 1.5:
            x_centre = i
            ax_plot = ax
        else:
            x_centre = i + 0.5
            ax_plot = ax2

        # Remove outliers
        plot_data = straight_cycles.loc[straight_cycles.status.eq(plot_info[0])][plot_info[1]]
        if plot_data.shape[0] < 5:
            continue
        range_IQ = np.diff(np.nanpercentile(plot_data, [25.0, 75.0]))[0]
        plot_data = plot_data.loc[
            plot_data.ge(np.nanpercentile(plot_data, 25.0) - 1.5 * range_IQ)
            & plot_data.le(np.nanpercentile(plot_data, 75.0) + 1.5 * range_IQ)
        ]
        boxplot_levels = np.nanpercentile(plot_data, [0.0, 25.0, 50.0, 75.0, 100.0])

        # Plot upper and lower whiskers
        ax_plot.plot(
            [
                x_centre - 0.4,
                x_centre + 0.4,
                np.nan,
                x_centre - 0.4,
                x_centre + 0.4,
                np.nan,
                x_centre,
                x_centre,
                np.nan,
                x_centre,
                x_centre,
            ],
            [
                boxplot_levels[0],
                boxplot_levels[0],
                np.nan,
                boxplot_levels[-1],
                boxplot_levels[-1],
                np.nan,
                boxplot_levels[0],
                boxplot_levels[1],
                np.nan,
                boxplot_levels[-1],
                boxplot_levels[-2],
            ],
            zorder=3.0,
            c="k",
        )

        # Plot inter-quartile box
        ax_plot.plot(
            [x_centre - 0.4, x_centre - 0.4, x_centre + 0.4, x_centre + 0.4, x_centre - 0.4],
            [
                boxplot_levels[1],
                boxplot_levels[3],
                boxplot_levels[3],
                boxplot_levels[1],
                boxplot_levels[1],
            ],
            zorder=2.0,
            c="C0",
        )

        # Plot median bar
        ax_plot.plot(
            [
                x_centre - 0.4,
                x_centre + 0.4,
            ],
            [
                boxplot_levels[2],
                boxplot_levels[2],
            ],
            zorder=1.0,
            c="C1",
        )


# ------------------------------------------------------------------------------
def plot_times_distances_on_axes(
    ax,
    straight_cycles,
):

    # Put something in place
    ax2 = ax.twinx()
    ax.set_title("Cycle Distances and Times")
    ax.set_xticks([0.0, 1.0, 2.5, 3.5])
    ax.set_xticklabels(
        [
            "Push\n(m)",
            "Return\n(m)",
            "Push\n(s)",
            "Return\n(s)",
        ]
    )
    ax.set_ylabel("Distance (m)")
    ax2.set_ylabel("Time (s)")

    # Any positions?
    if straight_cycles.shape[0] == 0:
        return

    # Process times and distances
    xticklabels = []
    for i, plot_info in enumerate(
        [
            ["push", "distance"],
            ["return", "distance"],
            ["push", "time"],
            ["return", "time"],
        ]
    ):

        # Distance or time plot?
        if i < 1.5:
            x_centre = i
            ax_plot = ax
        else:
            x_centre = i + 0.5
            ax_plot = ax2

        # Remove outliers
        plot_data = straight_cycles.loc[straight_cycles.status.eq(plot_info[0])][plot_info[1]]
        if plot_data.shape[0] < 5:
            continue
        range_IQ = np.diff(np.nanpercentile(plot_data, [25.0, 75.0]))[0]
        plot_data = plot_data.loc[
            plot_data.ge(np.nanpercentile(plot_data, 25.0) - 1.5 * range_IQ)
            & plot_data.le(np.nanpercentile(plot_data, 75.0) + 1.5 * range_IQ)
        ]
        boxplot_levels = np.nanpercentile(plot_data, [0.0, 25.0, 50.0, 75.0, 100.0])

        # Plot upper and lower whiskers
        ax_plot.plot(
            [
                x_centre - 0.3,
                x_centre + 0.3,
                np.nan,
                x_centre - 0.3,
                x_centre + 0.3,
                np.nan,
                x_centre,
                x_centre,
                np.nan,
                x_centre,
                x_centre,
            ],
            [
                boxplot_levels[0],
                boxplot_levels[0],
                np.nan,
                boxplot_levels[-1],
                boxplot_levels[-1],
                np.nan,
                boxplot_levels[0],
                boxplot_levels[1],
                np.nan,
                boxplot_levels[-1],
                boxplot_levels[-2],
            ],
            zorder=3.0,
            c="k",
        )

        # Plot inter-quartile box
        ax_plot.plot(
            [x_centre - 0.4, x_centre - 0.4, x_centre + 0.4, x_centre + 0.4, x_centre - 0.4],
            [
                boxplot_levels[1],
                boxplot_levels[3],
                boxplot_levels[3],
                boxplot_levels[1],
                boxplot_levels[1],
            ],
            zorder=2.0,
            c="C0",
        )

        # Plot median bar
        ax_plot.plot(
            [
                x_centre - 0.4,
                x_centre + 0.4,
            ],
            [
                boxplot_levels[2],
                boxplot_levels[2],
            ],
            zorder=1.0,
            c="C1",
        )

        # Add to xticklabels
        xticklabels.append(
            plot_info[0][0].upper()
            + plot_info[0][1:]
            + "\n"
            + ("(%0.0fm)" % boxplot_levels[2] if plot_info[1] == "distance" else "(%0.0fs)" % boxplot_levels[2])
        )

    # Fix xticklabels
    ax.set_xticklabels(xticklabels)


# ------------------------------------------------------------------------------
def plot_grades_map_on_axes(
    ax,
    positions,
    positions_full,
    plot_title,
    grid_size=10.0,
    grade_bounds=[-26.0, 26.0],
    cmap="jet",
):

    # Any positions?
    if positions.shape[0] == 0:
        ax.set_title("%s\n" % plot_title)
        return

    # Grade pivot table
    values_E = np.arange(
        positions.grid_E.min(),
        positions.grid_E.max() + 0.1,
        grid_size,
    )
    values_N = np.arange(
        positions.grid_N.min(),
        positions.grid_N.max() + 0.1,
        grid_size,
    )
    grade_pivot = pd.concat(
        (
            positions[["grid_E", "grid_N", "grade"]],
            pd.DataFrame(
                np.vstack(
                    (
                        np.vstack(
                            (
                                values_E,
                                np.full(values_E.shape[0], values_N[0]),
                                np.full(values_E.shape[0], np.nan),
                            )
                        ).T,
                        np.vstack(
                            (
                                np.full(values_N.shape[0], values_E[0]),
                                values_N,
                                np.full(values_N.shape[0], np.nan),
                            )
                        ).T,
                    )
                ),
                columns=["grid_E", "grid_N", "grade"],
            ),
        ),
        ignore_index=True,
    ).pivot_table(index="grid_N", columns="grid_E", values="grade", aggfunc="median")

    # Plot grades
    plt.imshow(
        np.flipud(grade_pivot.values),
        extent=(
            values_E[0] - 0.5 * grid_size,
            values_E[-1] + 0.5 * grid_size,
            values_N[0] - 0.5 * grid_size,
            values_N[-1] + 0.5 * grid_size,
        ),
        zorder=1.0,
        cmap=cmap,
        vmin=grade_bounds[0],
        vmax=grade_bounds[1],
    )
    plt.colorbar()

    # Adjust extents
    data_extents = np.array(
        [
            positions_full.easting.min()
            - dozer_map_padding_ratio * (positions_full.easting.max() - positions_full.easting.min()),
            positions_full.easting.max()
            + dozer_map_padding_ratio * (positions_full.easting.max() - positions_full.easting.min()),
            positions_full.northing.min()
            - dozer_map_padding_ratio * (positions_full.northing.max() - positions_full.northing.min()),
            positions_full.northing.max()
            + dozer_map_padding_ratio * (positions_full.northing.max() - positions_full.northing.min()),
        ]
    )
    plt.gca().set_xlim(data_extents[0:2])
    plt.gca().set_ylim(data_extents[2:])

    # Add labels
    ax.set_title("%s\n" % plot_title)
    ax.set_xlabel("Eastings (m)")
    ax.set_ylabel("Northings (m)")


# ------------------------------------------------------------------------------
def plot_speed_map_on_axes(
    ax, positions, positions_full, plot_title, grid_size=10.0, speed_bounds=[0.0, 12.0], cmap="jet"
):

    # Any positions?
    if positions.shape[0] == 0:
        ax.set_title("%s\n" % plot_title)
        return

    # Speed pivot table
    values_E = np.arange(
        positions.grid_E.min(),
        positions.grid_E.max() + 0.1,
        grid_size,
    )
    values_N = np.arange(
        positions.grid_N.min(),
        positions.grid_N.max() + 0.1,
        grid_size,
    )
    speed_pivot = pd.concat(
        (
            positions[["grid_E", "grid_N", "speed"]],
            pd.DataFrame(
                np.vstack(
                    (
                        np.vstack(
                            (
                                values_E,
                                np.full(values_E.shape[0], values_N[0]),
                                np.full(values_E.shape[0], np.nan),
                            )
                        ).T,
                        np.vstack(
                            (
                                np.full(values_N.shape[0], values_E[0]),
                                values_N,
                                np.full(values_N.shape[0], np.nan),
                            )
                        ).T,
                    )
                ),
                columns=["grid_E", "grid_N", "speed"],
            ),
        ),
        ignore_index=True,
    ).pivot_table(index="grid_N", columns="grid_E", values="speed", aggfunc="median")

    # Plot speeds
    plt.imshow(
        np.flipud(speed_pivot.values),
        extent=(
            values_E[0] - 0.5 * grid_size,
            values_E[-1] + 0.5 * grid_size,
            values_N[0] - 0.5 * grid_size,
            values_N[-1] + 0.5 * grid_size,
        ),
        zorder=1.0,
        cmap=cmap,
        vmin=speed_bounds[0],
        vmax=speed_bounds[1],
    )
    plt.colorbar()

    # Adjust extents
    data_extents = np.array(
        [
            positions_full.easting.min()
            - dozer_map_padding_ratio * (positions_full.easting.max() - positions_full.easting.min()),
            positions_full.easting.max()
            + dozer_map_padding_ratio * (positions_full.easting.max() - positions_full.easting.min()),
            positions_full.northing.min()
            - dozer_map_padding_ratio * (positions_full.northing.max() - positions_full.northing.min()),
            positions_full.northing.max()
            + dozer_map_padding_ratio * (positions_full.northing.max() - positions_full.northing.min()),
        ]
    )
    plt.gca().set_xlim(data_extents[0:2])
    plt.gca().set_ylim(data_extents[2:])

    # Add labels
    ax.set_title("%s\n" % plot_title)
    ax.set_xlabel("Eastings (m)")
    ax.set_ylabel("Northings (m)")


# ------------------------------------------------------------------------------
def plot_grade_profile_on_axes(
    ax,
    positions,
    plot_title,
):

    # Any positions?
    if positions.shape[0] == 0:
        ax.set_title("%s" % plot_title)
        return

    # Plot grades
    distance_grades = 10.0 * np.floor(0.99999 * positions.push_percent / 10.0) + 5.0
    distance_grades = positions.groupby(distance_grades).grade.median()
    plt.bar(
        distance_grades.index,
        distance_grades.values,
        width=8.0,
    )

    # Add labels
    ax.set_title("%s\n" % plot_title)
    ax.set_xlabel("Distance along push (%)")
    ax.set_ylabel("Median Grade (%)")


# ------------------------------------------------------------------------------
def plot_speed_profile_on_axes(
    ax,
    positions,
    plot_title,
):

    # Any positions?
    if positions.shape[0] == 0:
        ax.set_title("%s" % plot_title)
        return

    # Plot speeds
    positions.speed.hist(ax=ax, bins=np.arange(-0.25, 12.3, 0.5))

    # Add labels
    ax.set_title("%s" % plot_title)
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("GPS Readings (count)")


# ------------------------------------------------------------------------------
def plot_profile_on_axes(
    ax,
    data_series,
    plot_title,
    plot_xlabel,
    bins,
):

    # Any positions?
    if data_series.shape[0] == 0:
        ax.set_title(plot_title)
        return

    # Plot speeds
    data_series.hist(ax=ax, bins=bins)

    # Add labels
    ax.set_title(plot_title)
    ax.set_xlabel(plot_xlabel)
    ax.set_ylabel("GPS Readings (count)")


# ------------------------------------------------------------------------------
def plot_locale_page_OLD(
    site,
    shift_description,
    time_start,
    time_stop,
    positions,
    straight_cycles,
    area_id,
    grid_size=10.0,
):

    # Split shift into thirds
    time_bounds = [
        time_start,
        time_start + (time_stop - time_start) / 3.0,
        time_start + 2.0 * (time_stop - time_start) / 3.0,
        time_stop,
    ]
    time_ranges = [[time_bounds[idx - 1], time_bounds[idx]] for idx in range(1, len(time_bounds))]

    # Initialise figure
    fig = plt.figure(
        figsize=(dims_figure_size[0] / 25.4, dims_figure_size[1] / 25.4),
    )
    fig.clf()

    # Add easting and northing grid values
    positions = positions.copy()
    positions.loc[:, "grid_E"] = grid_size * (positions.easting / grid_size).round()
    positions.loc[:, "grid_N"] = grid_size * (positions.northing / grid_size).round()

    # Isolate cycle state positions
    positions_push = positions.loc[positions.straight_type.eq("push")]

    # Plot grades
    ax = fig.add_subplot(421)
    plot_grades_map_on_axes(
        ax,
        positions_push,
        positions_push,
        "Grades (full shift)",
    )
    for i, time_range in enumerate(time_ranges):

        # Plot grade map
        ax = fig.add_subplot(4, 2, 3 + i * 2)
        plot_grades_map_on_axes(
            ax,
            positions_push.loc[positions_push.datetime.ge(time_range[0]) & positions_push.datetime.le(time_range[1])],
            positions_push,
            "Grades (%s - %s)"
            % (
                time_range[0].strftime("%H:%M"),
                time_range[1].strftime("%H:%M"),
            ),
        )

        # Plot grade profile
        ax = fig.add_subplot(4, 2, 4 + i * 2)
        plot_grade_profile_on_axes(
            ax,
            positions_push.loc[positions_push.datetime.ge(time_range[0]) & positions_push.datetime.le(time_range[1])],
            "Grade Profile (%s - %s)"
            % (
                time_range[0].strftime("%H:%M"),
                time_range[1].strftime("%H:%M"),
            ),
        )

    # Plot distances and times
    ax = fig.add_subplot(422)
    plot_times_distances_on_axes(
        ax,
        straight_cycles,
    )

    # Tighten plot
    plt.tight_layout(
        pad=1.0,
        h_pad=1.0,
        w_pad=1.0,
        rect=[
            dims_subplots[0] / dims_figure_size[0],
            dims_subplots[2] / dims_figure_size[1],
            dims_subplots[1] / dims_figure_size[0],
            dims_subplots[3] / dims_figure_size[1],
        ],
    )

    # Add title
    title_string = "Dozer Push %s %s: Area %d" % (
        site,
        shift_description,
        area_id,
    )
    add_title_string(fig, title_string, dims_axes_title[0], 16.0)
    title_string = compile_cycle_type_stat_string(positions)
    add_title_string(fig, title_string, dims_axes_title[1], 13.0)

    # Return figure
    return fig


# ------------------------------------------------------------------------------
def plot_dozer_page_OLD(
    site,
    shift_description,
    time_start,
    time_stop,
    equipment,
    positions,
    straight_cycles,
    background_image,
):

    # Split shift into thirds
    time_bounds = [
        time_start,
        time_start + (time_stop - time_start) / 3.0,
        time_start + 2.0 * (time_stop - time_start) / 3.0,
        time_stop,
    ]
    time_ranges = [[time_bounds[idx - 1], time_bounds[idx]] for idx in range(1, len(time_bounds))]

    # Initialise figure
    fig = plt.figure(
        figsize=(dims_figure_size[0] / 25.4, dims_figure_size[1] / 25.4),
    )
    fig.clf()

    # Isolate cycle state positions
    positions_push = positions.loc[positions.straight_type.eq("push")]
    positions_return = positions.loc[positions.straight_type.eq("return")]

    # Plot grades
    ax = fig.add_subplot(211)
    plot_map_to_axis(
        ax,
        positions,
        background_image,
    )

    # Plot distances and times
    ax = fig.add_subplot(425)
    plot_times_distances_on_axes(
        ax,
        straight_cycles,
    )

    # Plot grade profile
    ax = fig.add_subplot(426)
    plot_grade_profile_on_axes(ax, positions_push, "Grade Profile (full shift)")

    # Plot speed profiles
    ax = fig.add_subplot(427)
    plot_speed_profile_on_axes(ax, positions_push, "Push Speeds")
    ax = fig.add_subplot(428)
    plot_speed_profile_on_axes(ax, positions_return, "Return Speeds")

    # Tighten plot
    plt.tight_layout(
        pad=1.0,
        h_pad=1.0,
        w_pad=1.0,
        rect=[
            dims_subplots[0] / dims_figure_size[0],
            dims_subplots[2] / dims_figure_size[1],
            dims_subplots[1] / dims_figure_size[0],
            dims_subplots[3] / dims_figure_size[1],
        ],
    )

    # Add title
    title_string = "Dozer Push %s %s: %s (%0.1fhrs)" % (
        site,
        shift_description,
        equipment,
        positions.duration_point.sum() / 3600.0,
    )
    add_title_string(fig, title_string, dims_axes_title[0], 16.0)
    title_string = compile_cycle_type_stat_string(positions)
    add_title_string(fig, title_string, dims_axes_title[1], 13.0)

    # Return figure
    return fig


# ------------------------------------------------------------------------------
def plot_details_page_1(
    title_suffix,
    site,
    shift_description,
    time_start,
    time_stop,
    positions,
    straight_cycles,
    background_image,
    grid_size=10.0,
):

    # Initialise figure
    fig = plt.figure(
        figsize=(dims_figure_size[0] / 25.4, dims_figure_size[1] / 25.4),
    )
    fig.clf()

    # Add easting and northing grid values
    positions = positions.copy()
    positions.loc[:, "grid_E"] = grid_size * (positions.easting / grid_size).round()
    positions.loc[:, "grid_N"] = grid_size * (positions.northing / grid_size).round()

    # Isolate cycle state positions
    positions_push = positions.loc[positions.straight_type.eq("push")]
    positions_return = positions.loc[positions.straight_type.eq("return")]

    # Plot paths
    ax = fig.add_subplot(211)
    plot_map_to_axis(
        ax,
        positions,
        background_image,
    )

    # Plot grades
    ax = fig.add_subplot(425)
    plot_grades_map_on_axes(
        ax,
        positions_push,
        positions_push,
        "Grades (%)",
    )

    # Plot distances and times
    ax = fig.add_subplot(426)
    plot_times_distances_on_axes(
        ax,
        straight_cycles,
    )

    # Plot grades
    ax = fig.add_subplot(427)
    plot_speed_map_on_axes(
        ax,
        positions_push,
        positions_push,
        "Speeds Push (km/h)",
        speed_bounds=[0.0, 6.0],
    )
    ax = fig.add_subplot(428)
    plot_speed_map_on_axes(
        ax,
        positions_return,
        positions_return,
        "Speeds Return (km/h)",
        speed_bounds=[0.0, 12.0],
    )

    # Tighten plot
    plt.tight_layout(
        pad=1.0,
        h_pad=1.0,
        w_pad=1.0,
        rect=[
            dims_subplots[0] / dims_figure_size[0],
            dims_subplots[2] / dims_figure_size[1],
            dims_subplots[1] / dims_figure_size[0],
            dims_subplots[3] / dims_figure_size[1],
        ],
    )

    # Add title
    title_string = "Dozer Push %s %s: %s" % (
        site,
        shift_description,
        title_suffix,
    )
    add_title_string(fig, title_string, dims_axes_title[0], 16.0)
    title_string = compile_cycle_type_stat_string(positions)
    add_title_string(fig, title_string, dims_axes_title[1], 13.0)

    # Return figure
    return fig


# ------------------------------------------------------------------------------
def plot_details_page_2(
    title_suffix,
    site,
    shift_description,
    time_start,
    time_stop,
    positions,
    straight_cycles,
):

    # Initialise figure
    fig = plt.figure(
        figsize=(dims_figure_size[0] / 25.4, dims_figure_size[1] / 25.4),
    )
    fig.clf()

    # Isolate cycle state positions
    positions_push = positions.loc[positions.straight_type.eq("push")]
    positions_return = positions.loc[positions.straight_type.eq("return")]
    cycles_push = straight_cycles.loc[straight_cycles.status.eq("push")]
    cycles_return = straight_cycles.loc[straight_cycles.status.eq("return")]

    # Plot distances
    ax = fig.add_subplot(421)
    plot_profile_on_axes(
        ax,
        cycles_push.distance,
        "Push Distances",
        "Distance (m)",
        np.arange(0.0, 251.0, 10.0),
    )
    ax = fig.add_subplot(422)
    plot_profile_on_axes(
        ax,
        cycles_return.distance,
        "Return Distances",
        "Distance (m)",
        np.arange(0.0, 251.0, 10.0),
    )

    # Plot grades
    ax = fig.add_subplot(423)
    plot_profile_on_axes(
        ax,
        positions_push.grade,
        "Push Grades",
        "Grade (%)",
        np.arange(-26.0, 26.1, 2.0),
    )
    ax = fig.add_subplot(424)
    plot_profile_on_axes(
        ax,
        positions_return.grade,
        "Return Grades",
        "Grade (%)",
        np.arange(-26.0, 26.1, 2.0),
    )

    # Plot speeds
    ax = fig.add_subplot(425)
    plot_profile_on_axes(
        ax,
        positions_push.speed,
        "Push Speeds",
        "Speed (km/h)",
        np.arange(0.0, 12.1, 0.5),
    )
    ax = fig.add_subplot(426)
    plot_profile_on_axes(
        ax,
        positions_return.speed,
        "Return Speeds",
        "Speed (km/h)",
        np.arange(0.0, 12.1, 0.5),
    )

    # Plot times
    ax = fig.add_subplot(427)
    plot_profile_on_axes(
        ax,
        cycles_push.time,
        "Push Times",
        "Time (s)",
        np.arange(0.0, 401.0, 20.0),
    )
    ax = fig.add_subplot(428)
    plot_profile_on_axes(
        ax,
        cycles_return.time,
        "Return Times",
        "Time (s)",
        np.arange(0.0, 401.0, 20.0),
    )

    # Tighten plot
    plt.tight_layout(
        pad=1.0,
        h_pad=1.0,
        w_pad=1.0,
        rect=[
            dims_subplots[0] / dims_figure_size[0],
            dims_subplots[2] / dims_figure_size[1],
            dims_subplots[1] / dims_figure_size[0],
            dims_subplots[3] / dims_figure_size[1],
        ],
    )

    # Add title
    title_string = "Dozer Push %s %s: %s" % (
        site,
        shift_description,
        title_suffix,
    )
    add_title_string(fig, title_string, dims_axes_title[0], 16.0)
    title_string = compile_cycle_type_stat_string(positions)
    add_title_string(fig, title_string, dims_axes_title[1], 13.0)

    # Return figure
    return fig


# ------------------------------------------------------------------------------
def compile_dozer_report(
    site,
    shift_description,
    time_start,
    time_stop,
    positions,
    straight_cycles,
    background_image,
):

    # Add percentage of push cycle to positions
    positions.loc[:, "push_percent"] = np.nan
    for straight_cycle in straight_cycles.loc[straight_cycles.status.eq("push")].itertuples():

        # Assign percentages
        indices = positions.index[straight_cycle.idx_start : straight_cycle.idx_stop]
        positions.loc[indices, "push_percent"] = (
            ((positions.loc[indices].easting.diff() ** 2.0 + positions.loc[indices].northing.diff() ** 2.0) ** 0.5)
            .cumsum()
            .fillna(0.0)
        )
        positions.loc[indices, "push_percent"] /= positions.loc[indices, "push_percent"].max()
        positions.loc[indices, "push_percent"] *= 100.0

    # Open report
    with PdfPages(
        "Dozer_Push_%s_%s.pdf"
        % (
            site,
            shift_description.split(" ")[0].replace("-", "") + "_" + shift_description.split(" ")[1][0],
        )
    ) as pdf:

        # Overview map
        fig = plot_overview_map(site, shift_description, positions, background_image)
        pdf.savefig(fig)

        # Process locales
        for locale in positions.groupby("straight_cluster"):

            # Generate page for locale
            title_sufix = "Area %d" % (locale[0] + 1)
            fig = plot_details_page_1(
                title_sufix,
                site,
                shift_description,
                time_start,
                time_stop,
                locale[1],
                straight_cycles.loc[straight_cycles.cluster.eq(locale[0])],
                background_image,
            )
            pdf.savefig(fig)
            fig = plot_details_page_2(
                title_sufix,
                site,
                shift_description,
                time_start,
                time_stop,
                locale[1],
                straight_cycles.loc[straight_cycles.cluster.eq(locale[0])],
            )
            pdf.savefig(fig)

        # Process dozers
        for dozer in positions.groupby("equipment"):

            # Generate page for locale
            title_sufix = "%s (%0.1fhrs)" % (dozer[0], dozer[1].duration_point.sum() / 3600.0)
            fig = plot_details_page_1(
                title_sufix,
                site,
                shift_description,
                time_start,
                time_stop,
                dozer[1],
                straight_cycles.loc[straight_cycles.equipment.eq(dozer[0])],
                background_image,
            )
            pdf.savefig(fig)
            fig = plot_details_page_2(
                title_sufix,
                site,
                shift_description,
                time_start,
                time_stop,
                dozer[1],
                straight_cycles.loc[straight_cycles.equipment.eq(dozer[0])],
            )
            pdf.savefig(fig)

    # Close all plots
    plt.close("all")


######################################################################################################
# joyplots
# from joypy import joyplot
# Isolate cycle state positions
# positions_push = positions.loc[positions.straight_type.eq("push")]
# positions_return = positions.loc[positions.straight_type.eq("return")]
# cycles_push = straight_cycles.loc[straight_cycles.status.eq("push")]
# cycles_return = straight_cycles.loc[straight_cycles.status.eq("return")]

# # Push

# fig, axes = joyplot(
#     cycles_push,
#     by="equipment",
#     column=["distance"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Push Distances",
# )
# means = []
# medians = []
# stddevs = []
# for dozer in cycles_push.equipment.unique():
#     means.append(cycles_push[cycles_push.equipment.eq(dozer)]["distance"].mean())
#     medians.append(cycles_push[cycles_push.equipment.eq(dozer)]["distance"].median())
#     stddevs.append(cycles_push[cycles_push.equipment.eq(dozer)]["distance"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}m, M={round(medians[0],1)}m,\n\u03C3={round(stddevs[0],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}m, M={round(medians[1],1)}m,\n\u03C3={round(stddevs[1],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}m, M={round(medians[2],1)}m,\n\u03C3={round(stddevs[2],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}m, M={round(medians[3],1)}m,\n\u03C3={round(stddevs[3],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}m, M={round(medians[4],1)}m,\n\u03C3={round(stddevs[4],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}m, M={round(medians[5],1)}m,\n\u03C3={round(stddevs[5],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}m, M={round(medians[6],1)}m,\n\u03C3={round(stddevs[6],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}m, M={round(medians[7],1)}m,\n\u03C3={round(stddevs[7],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# plt.xlabel("Distance (m)")
# plt.show()

# fig, axes = joyplot(
#     cycles_push,
#     by="equipment",
#     column=["time"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Push Times",
#     x_range=[20.0, 801.0],
# )
# means = []
# medians = []
# stddevs = []
# for dozer in cycles_push.equipment.unique():
#     means.append(cycles_push[cycles_push.equipment.eq(dozer)]["time"].mean())
#     medians.append(cycles_push[cycles_push.equipment.eq(dozer)]["time"].median())
#     stddevs.append(cycles_push[cycles_push.equipment.eq(dozer)]["time"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}secs, M={round(medians[0],1)}secs,\n\u03C3={round(stddevs[0],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}secs, M={round(medians[1],1)}secs,\n\u03C3={round(stddevs[1],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}secs, M={round(medians[2],1)}secs,\n\u03C3={round(stddevs[2],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}secs, M={round(medians[3],1)}secs,\n\u03C3={round(stddevs[3],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}secs, M={round(medians[4],1)}secs,\n\u03C3={round(stddevs[4],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}secs, M={round(medians[5],1)}secs,\n\u03C3={round(stddevs[5],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}secs, M={round(medians[6],1)}secs,\n\u03C3={round(stddevs[6],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}secs, M={round(medians[7],1)}secs,\n\u03C3={round(stddevs[7],1)}secs",
#     (670.0, 0.002),
#     fontsize=13,
# )
# plt.xlabel("Time (secs)")
# plt.show()

# fig, axes = joyplot(
#     positions_push,
#     by="equipment",
#     column=["grade"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Push Grades",
#     ylim="own",
# )
# means = []
# medians = []
# stddevs = []
# for dozer in positions_push.equipment.unique():
#     means.append(positions_push[positions_push.equipment.eq(dozer)]["grade"].mean())
#     medians.append(positions_push[positions_push.equipment.eq(dozer)]["grade"].median())
#     stddevs.append(positions_push[positions_push.equipment.eq(dozer)]["grade"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}%, M={round(medians[0],1)}%,\n\u03C3={round(stddevs[0],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}%, M={round(medians[1],1)}%,\n\u03C3={round(stddevs[1],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}%, M={round(medians[2],1)}%,\n\u03C3={round(stddevs[2],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# # axes[3].annotate(
# #     f"\u03BC={round(means[3],1)}%, M={round(medians[3],1)}%,\n\u03C3={round(stddevs[3],1)}%",
# #     (-30.0, 0.002),
# #     fontsize=13,
# # )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}%, M={round(medians[4],1)}%,\n\u03C3={round(stddevs[4],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}%, M={round(medians[5],1)}%,\n\u03C3={round(stddevs[5],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}%, M={round(medians[6],1)}%,\n\u03C3={round(stddevs[6],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}%, M={round(medians[7],1)}%,\n\u03C3={round(stddevs[7],1)}%",
#     (-30.0, 0.002),
#     fontsize=13,
# )
# plt.xlabel("Grade (%)")
# plt.show()

# fig, axes = joyplot(
#     positions_push,
#     by="equipment",
#     column=["speed"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Push Speed",
#     x_range=[0.0, 7.0],
# )
# means = []
# medians = []
# stddevs = []
# for dozer in positions_push.equipment.unique():
#     means.append(positions_push[positions_push.equipment.eq(dozer)]["speed"].mean())
#     medians.append(positions_push[positions_push.equipment.eq(dozer)]["speed"].median())
#     stddevs.append(positions_push[positions_push.equipment.eq(dozer)]["speed"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}km/h, M={round(medians[0],1)}km/h, \u03C3={round(stddevs[0],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}km/h, M={round(medians[1],1)}km/h, \u03C3={round(stddevs[1],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}km/h, M={round(medians[2],1)}km/h, \u03C3={round(stddevs[2],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}km/h, M={round(medians[3],1)}km/h, \u03C3={round(stddevs[3],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}km/h, M={round(medians[4],1)}km/h, \u03C3={round(stddevs[4],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}km/h, M={round(medians[5],1)}km/h, \u03C3={round(stddevs[5],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}km/h, M={round(medians[6],1)}km/h, \u03C3={round(stddevs[6],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}km/h, M={round(medians[7],1)}km/h, \u03C3={round(stddevs[7],1)}km/h",
#     (5.0, 0.05),
#     fontsize=13,
# )
# plt.xlabel("Speed (km/h)")
# plt.show()

# # Return

# fig, axes = joyplot(
#     cycles_return,
#     by="equipment",
#     column=["distance"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Return Distances",
# )
# means = []
# medians = []
# stddevs = []
# for dozer in cycles_return.equipment.unique():
#     means.append(cycles_return[cycles_return.equipment.eq(dozer)]["distance"].mean())
#     medians.append(cycles_return[cycles_return.equipment.eq(dozer)]["distance"].median())
#     stddevs.append(cycles_return[cycles_return.equipment.eq(dozer)]["distance"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}m, M={round(medians[0],1)}m,\n\u03C3={round(stddevs[0],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}m, M={round(medians[1],1)}m,\n\u03C3={round(stddevs[1],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}m, M={round(medians[2],1)}m,\n\u03C3={round(stddevs[2],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}m, M={round(medians[3],1)}m,\n\u03C3={round(stddevs[3],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}m, M={round(medians[4],1)}m,\n\u03C3={round(stddevs[4],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}m, M={round(medians[5],1)}m,\n\u03C3={round(stddevs[5],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}m, M={round(medians[6],1)}m,\n\u03C3={round(stddevs[6],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}m, M={round(medians[7],1)}m,\n\u03C3={round(stddevs[7],1)}m",
#     (255.0, 0.002),
#     fontsize=13,
# )
# plt.xlabel("Distance (m)")
# plt.show()

# fig, axes = joyplot(
#     cycles_return[cycles_return.speed > 0],
#     by="equipment",
#     column=["time"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Return Times",
#     x_range=[-1.0, 601.0],
# )
# means = []
# medians = []
# stddevs = []
# for dozer in cycles_return.equipment.unique():
#     means.append(cycles_return[cycles_return.equipment.eq(dozer)]["time"].mean())
#     medians.append(cycles_return[cycles_return.equipment.eq(dozer)]["time"].median())
#     stddevs.append(cycles_return[cycles_return.equipment.eq(dozer)]["time"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}secs, M={round(medians[0],1)}secs,\n\u03C3={round(stddevs[0],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}secs, M={round(medians[1],1)}secs,\n\u03C3={round(stddevs[1],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}secs, M={round(medians[2],1)}secs,\n\u03C3={round(stddevs[2],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}secs, M={round(medians[3],1)}secs,\n\u03C3={round(stddevs[3],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}secs, M={round(medians[4],1)}secs,\n\u03C3={round(stddevs[4],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}secs, M={round(medians[5],1)}secs,\n\u03C3={round(stddevs[5],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}secs, M={round(medians[6],1)}secs,\n\u03C3={round(stddevs[6],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}secs, M={round(medians[7],1)}secs,\n\u03C3={round(stddevs[7],1)}secs",
#     (450.0, 0.002),
#     fontsize=13,
# )
# plt.xlabel("Time (secs)")
# plt.show()

# fig, axes = joyplot(
#     positions_return,
#     by="equipment",
#     column=["grade"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Return Grades",
#     ylim="own",
# )
# means = []
# medians = []
# stddevs = []
# for dozer in positions_return.equipment.unique():
#     means.append(positions_return[positions_return.equipment.eq(dozer)]["grade"].mean())
#     medians.append(positions_return[positions_return.equipment.eq(dozer)]["grade"].median())
#     stddevs.append(positions_return[positions_return.equipment.eq(dozer)]["grade"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}%, M={round(medians[0],1)}%,\n\u03C3={round(stddevs[0],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}%, M={round(medians[1],1)}%,\n\u03C3={round(stddevs[1],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}%, M={round(medians[2],1)}%,\n\u03C3={round(stddevs[2],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}%, M={round(medians[3],1)}%,\n\u03C3={round(stddevs[3],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}%, M={round(medians[4],1)}%,\n\u03C3={round(stddevs[4],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}%, M={round(medians[5],1)}%,\n\u03C3={round(stddevs[5],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}%, M={round(medians[6],1)}%,\n\u03C3={round(stddevs[6],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}%, M={round(medians[7],1)}%,\n\u03C3={round(stddevs[7],1)}%",
#     (20.0, 0.007),
#     fontsize=13,
# )
# plt.xlabel("Grade (%)")
# plt.show()

# fig, axes = joyplot(
#     positions_return,
#     by="equipment",
#     column=["speed"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Return Speed",
#     x_range=[0.0, 12.0],
# )
# means = []
# medians = []
# stddevs = []
# for dozer in positions_return.equipment.unique():
#     means.append(positions_return[positions_return.equipment.eq(dozer)]["speed"].mean())
#     medians.append(positions_return[positions_return.equipment.eq(dozer)]["speed"].median())
#     stddevs.append(positions_return[positions_return.equipment.eq(dozer)]["speed"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}km/h, M={round(medians[0],1)}km/h,\n\u03C3={round(stddevs[0],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}km/h, M={round(medians[1],1)}km/h,\n\u03C3={round(stddevs[1],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}km/h, M={round(medians[2],1)}km/h,\n\u03C3={round(stddevs[2],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}km/h, M={round(medians[3],1)}km/h,\n\u03C3={round(stddevs[3],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}km/h, M={round(medians[4],1)}km/h,\n\u03C3={round(stddevs[4],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}km/h, M={round(medians[5],1)}km/h,\n\u03C3={round(stddevs[5],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}km/h, M={round(medians[6],1)}km/h,\n\u03C3={round(stddevs[6],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}km/h, M={round(medians[7],1)}km/h,\n\u03C3={round(stddevs[7],1)}km/h",
#     (0.0, 0.08),
#     fontsize=13,
# )
# plt.xlabel("Speed (km/h)")
# plt.show()


# Correlation Matrices
# from string import ascii_letters
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# Return

# sns.set_theme(style="dark")

# # Compute the correlation matrix
# df = cycles_return[["distance", "time", "speed"]]
# df = df[df.time < 1000]
# corr = cycles_push[["distance", "time", "speed"]].corr()
# corr = cycles_push[["distance", "time", "speed"]].corr()
# corr = positions_return[["grade", "speed"]].corr()
# corr = positions_push[["grade", "speed"]].corr()

# # Generate a mask for the upper triangle
# mask = np.triu(corr)

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, annot = True, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
# plt.title("Dozer Push Statistics: Return Cycle Aggregate Parameters")


##################################################################

# # Productivity
# from joypy import joyplot
# import matplotlib.pyplot as plt
# # straight_cycles = straight_cycles[straight_cycles["Total_Cycle_Time (sec)"] > 158]
# fig, axes = joyplot(
#     straight_cycles,
#     by="Equipment_Code",
#     column=["Productivity (bcm/h)"],
#     fade=True,
#     title="VER 2023-02-07: Density Profiles of Dozer Productivity",
#     x_range=[0,1000]
# )
# means = []
# medians = []
# stddevs = []
# for dozer in straight_cycles.Equipment_Code.unique():
#     means.append(straight_cycles[straight_cycles.Equipment_Code.eq(dozer)]["Productivity (bcm/h)"].mean())
#     medians.append(straight_cycles[straight_cycles.Equipment_Code.eq(dozer)]["Productivity (bcm/h)"].median())
#     stddevs.append(straight_cycles[straight_cycles.Equipment_Code.eq(dozer)]["Productivity (bcm/h)"].std())
# axes[0].annotate(
#     f"\u03BC={round(means[0],1)}bcm/h, M={round(medians[0],1)}bcm/h,\n\u03C3={round(stddevs[0],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[1].annotate(
#     f"\u03BC={round(means[1],1)}bcm/h, M={round(medians[1],1)}bcm/h,\n\u03C3={round(stddevs[1],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[2].annotate(
#     f"\u03BC={round(means[2],1)}bcm/h, M={round(medians[2],1)}bcm/h,\n\u03C3={round(stddevs[2],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[3].annotate(
#     f"\u03BC={round(means[3],1)}bcm/h, M={round(medians[3],1)}bcm/h,\n\u03C3={round(stddevs[3],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[4].annotate(
#     f"\u03BC={round(means[4],1)}bcm/h, M={round(medians[4],1)}bcm/h,\n\u03C3={round(stddevs[4],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[5].annotate(
#     f"\u03BC={round(means[5],1)}bcm/h, M={round(medians[5],1)}bcm/h,\n\u03C3={round(stddevs[5],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[6].annotate(
#     f"\u03BC={round(means[6],1)}bcm/h, M={round(medians[6],1)}bcm/h,\n\u03C3={round(stddevs[6],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# axes[7].annotate(
#     f"\u03BC={round(means[7],1)}bcm/h, M={round(medians[7],1)}bcm/h,\n\u03C3={round(stddevs[7],1)}bcm/h",
#     (650.0, 0.001),
#     fontsize=13,
# )
# plt.xlabel("Productivity (bcm/h)")
# plt.show()


####################################################

# Plotting KPI over time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

positions_push = positions.loc[positions.straight_type.eq("push")]
positions_return = positions.loc[positions.straight_type.eq("return")]
cycles_push = straight_cycles.loc[straight_cycles.status.eq("push")]
cycles_return = straight_cycles.loc[straight_cycles.status.eq("return")]

symbol_dict = color_dict = {
    "DZ2156": "circle",
    "DZ2157": "diamond",
    "DZ2173": "cross",
    "DZ2174": "pentagon",
    "DZ2175": "triangle-up",
    "DZ8032": "star-diamond",
    "DZ8035": "diamond-wide",
    "DZ8036": "diamond-x",
}
color_dict = {
    "DZ2156": "blue",
    "DZ2157": "red",
    "DZ2173": "green",
    "DZ2174": "purple",
    "DZ2175": "orange",
    "DZ8032": "lightblue",
    "DZ8035": "pink",
    "DZ8036": "brown",
}

pos_push = positions_push.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)

dozers = pos_push.equipment.unique()
pos_push = pos_push.set_index("datetime")
pos_push = pos_push.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_push = pos_push.reset_index(names=["equipment", "datetime"])

fig = make_subplots(
    x_title="Local Datetime",
    shared_xaxes=True,
    shared_yaxes=True,
    rows=8,
    cols=1,
    specs=[
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
    ],
)

# First Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[0])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[0])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        line=go.scatter.Line(color="red"),
    ),
    row=1,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[0])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[0])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        line=go.scatter.Line(color="blue"),
    ),
    row=1,
    col=1,
    secondary_y=False,
)

# Second Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[1])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[1])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=2,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[1])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[1])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=2,
    col=1,
    secondary_y=False,
)

# Third Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[2])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[2])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=3,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[2])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[2])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=3,
    col=1,
    secondary_y=False,
)

# Fourth Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[3])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[3])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=4,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[3])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[3])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=4,
    col=1,
    secondary_y=False,
)

# Fifth Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[4])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[4])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=5,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[4])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[4])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=5,
    col=1,
    secondary_y=False,
)

# Sixth Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[5])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[5])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=6,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[5])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[5])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=6,
    col=1,
    secondary_y=False,
)

# Seventh Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[6])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[6])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=7,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[6])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[6])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=7,
    col=1,
    secondary_y=False,
)

# Eigth Subplot
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[7])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[7])].straight_speed,
        name=f"Push Speed",
        legendgroup="Push Speed",
        showlegend=False,
        line=go.scatter.Line(color="red"),
    ),
    row=8,
    col=1,
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=pos_push[pos_push.equipment.eq(dozers[7])].datetime,
        y=pos_push[pos_push.equipment.eq(dozers[7])].straight_distance,
        name=f"Push Distance",
        legendgroup="Push Distance",
        showlegend=False,
        line=go.scatter.Line(color="blue"),
    ),
    row=8,
    col=1,
    secondary_y=False,
)


# Add figure title
fig.update_layout(title_text="Hourly Mean Push Speed and Distance Across Shift")

subplot_titles = ["Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 5", "Plot 6", "Plot 7", "Plot 8"]
for i, col in enumerate(subplot_titles, start=0):
    fig.update_yaxes(title_text=f"<b>{dozers[i]}</b><br> Mean<br>Distance (m)", row=i + 1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Mean<br>Speed (km/h)", row=i + 1, col=1, secondary_y=True)


fig.update_yaxes(range=[0, 400], secondary_y=False)
fig.update_yaxes(range=[1, 4], secondary_y=True)

fig.show()


##########################################################
# speed and distance
symbol_dict = color_dict = {
    "DZ2156": "circle",
    "DZ2157": "diamond",
    "DZ2173": "cross",
    "DZ2174": "pentagon",
    "DZ2175": "triangle-up",
    "DZ8032": "star-diamond",
    "DZ8035": "diamond-wide",
    "DZ8036": "diamond-x",
}

pos_push = positions_push.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
pos_return = positions_return.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
dozers = pos_push.equipment.unique()
color_dict = {
    "DZ2156": "blue",
    "DZ2157": "red",
    "DZ2173": "green",
    "DZ2174": "purple",
    "DZ2175": "orange",
    "DZ8032": "lightblue",
    "DZ8035": "pink",
    "DZ8036": "brown",
}
pos_push = pos_push.set_index("datetime")
pos_push = pos_push.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_push = pos_push.reset_index(names=["equipment", "datetime"])

pos_return = pos_return.set_index("datetime")
pos_return = pos_return.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_return = pos_return.reset_index(names=["equipment", "datetime"])

fig = make_subplots(
    x_title="Time of Push Cycle Aggregate",
    shared_xaxes=True,
    shared_yaxes=True,
    rows=4,
    cols=1,
)

# First Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_speed,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=1,
        col=1,
    )

# Second Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_distance,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=2,
        col=1,
    )


# Third Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_grade,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=3,
        col=1,
    )


# Fourth Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_speed,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=4,
        col=1,
    )


fig.update_layout(title_text="Hourly Mean Speed and Distance Across Shift")
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Speed (km/h)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Distance (m)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Speed (km/h)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Distance (m)", row=4, col=1, secondary_y=False)
fig.update_layout(
    font=dict(size=16),
)
fig.update_traces(
    marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
)
fig.show()


##########################################################
# Grade
symbol_dict = color_dict = {
    "DZ2156": "circle",
    "DZ2157": "diamond",
    "DZ2173": "cross",
    "DZ2174": "pentagon",
    "DZ2175": "triangle-up",
    "DZ8032": "star-diamond",
    "DZ8035": "diamond-wide",
    "DZ8036": "diamond-x",
}

pos_push = positions_push.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
pos_return = positions_return.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
dozers = pos_push.equipment.unique()
color_dict = {
    "DZ2156": "blue",
    "DZ2157": "red",
    "DZ2173": "green",
    "DZ2174": "purple",
    "DZ2175": "orange",
    "DZ8032": "lightblue",
    "DZ8035": "pink",
    "DZ8036": "brown",
}
pos_push = pos_push.set_index("datetime")
pos_push = pos_push.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_push = pos_push.reset_index(names=["equipment", "datetime"])

pos_return = pos_return.set_index("datetime")
pos_return = pos_return.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_return = pos_return.reset_index(names=["equipment", "datetime"])

fig = make_subplots(
    x_title="Time of Push Cycle Aggregate",
    shared_xaxes=True,
    shared_yaxes=True,
    rows=4,
    cols=1,
)

# First Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_grade,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=1,
        col=1,
    )

# Second Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_distance,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=2,
        col=1,
    )


# Third Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_grade,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=3,
        col=1,
    )


# Fourth Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_distance,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=4,
        col=1,
    )


fig.update_layout(title_text="Hourly Mean Grade and Distance Across Shift")
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Grade (%)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Distance (m)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Grade (%)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Distance (m)", row=4, col=1, secondary_y=False)
fig.update_layout(
    font=dict(size=16),
)
fig.update_traces(
    marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
)
fig.show()

###################################################

# Grade
symbol_dict = color_dict = {
    "DZ2156": "circle",
    "DZ2157": "diamond",
    "DZ2173": "cross",
    "DZ2174": "pentagon",
    "DZ2175": "triangle-up",
    "DZ8032": "star-diamond",
    "DZ8035": "diamond-wide",
    "DZ8036": "diamond-x",
}
pos_push = positions_push.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
pos_return = positions_return.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
dozers = pos_push.equipment.unique()
color_dict = {
    "DZ2156": "blue",
    "DZ2157": "red",
    "DZ2173": "green",
    "DZ2174": "purple",
    "DZ2175": "orange",
    "DZ8032": "lightblue",
    "DZ8035": "pink",
    "DZ8036": "brown",
}
pos_push = pos_push.set_index("datetime")
pos_push = pos_push.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_push = pos_push.reset_index(names=["equipment", "datetime"])

pos_return = pos_return.set_index("datetime")
pos_return = pos_return.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_return = pos_return.reset_index(names=["equipment", "datetime"])

fig = make_subplots(
    x_title="Time of Push Cycle Aggregate",
    shared_xaxes=True,
    shared_yaxes=True,
    rows=4,
    cols=1,
)

# First Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_grade,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=1,
        col=1,
    )

# Second Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_speed,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=2,
        col=1,
    )


# Third Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_grade,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=3,
        col=1,
    )


# Fourth Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_speed,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=4,
        col=1,
    )

fig.update_layout(title_text="Hourly Mean Grade and Speed Across Shift")
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Grade (%)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Speed (km/h)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Grade (%)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Speed (km/h)", row=4, col=1, secondary_y=False)
fig.update_layout(
    font=dict(size=16),
)
fig.update_traces(
    marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
)
fig.show()

###################################################

# Grade
pos_push = positions_push.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
pos_return = positions_return.drop_duplicates(
    [
        "straight_start",
        "straight_stop",
        "straight_track",
        "straight_distance",
        "straight_speed",
        "straight_type",
        "straight_grade",
        "straight_cluster",
    ]
)
dozers = pos_push.equipment.unique()
color_dict = {
    "DZ2156": "blue",
    "DZ2157": "red",
    "DZ2173": "green",
    "DZ2174": "purple",
    "DZ2175": "orange",
    "DZ8032": "lightblue",
    "DZ8035": "pink",
    "DZ8036": "brown",
}
pos_push = pos_push.set_index("datetime")
pos_push = pos_push.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_push = pos_push.reset_index(names=["equipment", "datetime"])

pos_return = pos_return.set_index("datetime")
pos_return = pos_return.groupby(["equipment", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
pos_return = pos_return.reset_index(names=["equipment", "datetime"])

fig = make_subplots(
    x_title="Local Datetime",
    shared_xaxes=True,
    shared_yaxes=True,
    rows=4,
    cols=1,
)

# First Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_grade,
            name=dozer,
            legendgroup=dozer,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=1,
        col=1,
    )

# Second Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_push[pos_push.equipment.eq(dozer)].datetime,
            y=pos_push[pos_push.equipment.eq(dozer)].straight_speed,
            name=dozer,
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=2,
        col=1,
    )


# Third Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_grade,
            name=dozer,
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=3,
        col=1,
    )


# Fourth Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=pos_return[pos_return.equipment.eq(dozer)].datetime,
            y=pos_return[pos_return.equipment.eq(dozer)].straight_speed,
            name=dozer,
            legendgroup=dozer,
            showlegend=False,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=4,
        col=1,
    )

fig.update_layout(title_text="Hourly Mean Grade and Distance Across Shift", xaxis_title="Local Datetime")
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Grade (%)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Push</b><br> Mean<br>Speed (km/h)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Grade (%)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"<b>Return</b><br> Mean<br>Speed (km/h)", row=4, col=1, secondary_y=False)
fig.update_layout(font=dict(size=16))
fig.update_traces(
    marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
)
fig.show()

##########################################
straight_cycles["Local_Datetime"] = positions.loc[straight_cycles.Straight_Index_Start]["Local_Datetime"].reset_index(
    drop=True
)

import plotly.graph_objects as go

dozers = straight_cycles.Equipment_Code.unique()
symbol_dict = color_dict = {
    "DZ2156": "circle",
    "DZ2157": "diamond",
    "DZ2173": "cross",
    "DZ2174": "pentagon",
    "DZ2175": "triangle-up",
    "DZ8032": "star-diamond",
    "DZ8035": "diamond-wide",
    "DZ8036": "diamond-x",
}
color_dict = {
    "DZ2156": "#000000",
    "DZ2157": "#000000",
    "DZ2173": "#000000",
    "DZ2174": "#000000",
    "DZ2175": "#000000",
    "DZ8032": "#000000",
    "DZ8035": "#000000",
    "DZ8036": "#000000",
}
color_list = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
]
# color_list = ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"]
straight_cycles_copy = straight_cycles[straight_cycles["Total_Cycle_Time (sec)"] > 200]

cycles = straight_cycles_copy.set_index("Local_Datetime")
# Outlier removal
# cycles[cycles.Equipment_Code.eq(dozers[0])] = cycles[
#     cycles.Equipment_Code.eq(dozers[0])
#     & (cycles["Total_Cycle_Time (sec)"] < 559)
#     & (cycles["Total_Cycle_Time (sec)"] > 148)
# ]
# cycles[cycles.Equipment_Code.eq(dozers[1])] = cycles[
#     cycles.Equipment_Code.eq(dozers[1]) & (cycles["Total_Cycle_Time (sec)"] > 148)
# ]
# cycles[cycles.Equipment_Code.eq(dozers[2])] = cycles[
#     cycles.Equipment_Code.eq(dozers[2])
#     & (cycles["Total_Cycle_Time (sec)"] < 363)
#     & (cycles["Total_Cycle_Time (sec)"] > 158)
# ]
# cycles[cycles.Equipment_Code.eq(dozers[3])] = cycles[
#     cycles.Equipment_Code.eq(dozers[3]) & (cycles["Total_Cycle_Time (sec)"] < 661)
# ]
# cycles[cycles.Equipment_Code.eq(dozers[4])] = cycles[
#     cycles.Equipment_Code.eq(dozers[4])
#     & (cycles["Total_Cycle_Time (sec)"] < 705)
#     & (cycles["Total_Cycle_Time (sec)"] > 270)
# ]
# cycles[cycles.Equipment_Code.eq(dozers[6])] = cycles[
#     cycles.Equipment_Code.eq(dozers[6]) & (cycles["Total_Cycle_Time (sec)"] < 725)
# ]
# cycles[cycles.Equipment_Code.eq(dozers[7])] = cycles[
#     cycles.Equipment_Code.eq(dozers[7]) & (cycles["Total_Cycle_Time (sec)"] < 738)
# ]

cycles = cycles.groupby(["Equipment_Code", pd.Grouper(freq="60Min", base=30, label="right")]).mean()
cycles = cycles.reset_index(names=["Equipment_Code", "Local_Datetime"])

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[0])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[0])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[0]]),
        line_color=color_list[0],
        name=dozers[0],
    )
)
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[1])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[1])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[1]]),
        line_color=color_list[1],
        name=dozers[1],
    )
)
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[2])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[2])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[2]]),
        line_color=color_list[2],
        name=dozers[2],
    )
)
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[3])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[3])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[3]]),
        line_color=color_list[3],
        name=dozers[3],
    )
)

fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[4])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[4])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[4]]),
        line_color=color_list[4],
        name=dozers[4],
    )
)
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[5])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[5])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[5]]),
        line_color=color_list[5],
        name=dozers[5],
    )
)
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[6])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[6])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[6]]),
        line_color=color_list[6],
        name=dozers[6],
    )
)
fig.add_trace(
    go.Scatter(
        x=cycles[cycles.Equipment_Code.eq(dozers[7])].Local_Datetime.dropna(),
        y=cycles[cycles.Equipment_Code.eq(dozers[7])]["Productivity (bcm/h)"].dropna(),
        fill=None,
        marker=dict(symbol=symbol_dict[dozers[7]]),
        line_color=color_list[7],
        name=dozers[7],
    )
)
fig.update_layout(template="ggplot2")
fig.update_layout(
    title="Mean Hourly Productivity (BCM/H) Per Dozer",
    xaxis_title="Time of Push Cycle Aggregate",
    yaxis_title="Mean Productivity (BCM/H)",
    font=dict(size=18),
)
fig.update_traces(
    marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
)

fig.show()


# Get Sense of Spread of total time for each dozer and filter out outliers
import plotly.express as px

fig = px.box(straight_cycles, x="Equipment_Code", y="Total_Cycle_Time (sec)")
fig.show()


# Just plot prod v time
straight_cycles["Local_Datetime"] = positions.loc[straight_cycles.Straight_Index_Start]["Local_Datetime"].reset_index(
    drop=True
)

import plotly.graph_objects as go

dozers = straight_cycles.Equipment_Code.unique()
color_dict = {
    "DZ2156": "#b2182b",
    "DZ2157": "#d6604d",
    "DZ2173": "#f4a582",
    "DZ2174": "#de77ae",
    "DZ2175": "#66bd63",
    "DZ8032": "#92c5de",
    "DZ8035": "#4393c3",
    "DZ8036": "#2166ac",
}
color_list = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
]
cycles = straight_cycles
# Outlier removal
cycles[cycles.Equipment_Code.eq(dozers[0])] = cycles[
    cycles.Equipment_Code.eq(dozers[0]) & (cycles["Productivity (bcm/h)"] < 545)
]
cycles[cycles.Equipment_Code.eq(dozers[1])] = cycles[
    cycles.Equipment_Code.eq(dozers[1]) & (cycles["Productivity (bcm/h)"] < 800)
]
cycles[cycles.Equipment_Code.eq(dozers[2])] = cycles[
    cycles.Equipment_Code.eq(dozers[2]) & (cycles["Productivity (bcm/h)"] < 629)
]
cycles[cycles.Equipment_Code.eq(dozers[6])] = cycles[
    cycles.Equipment_Code.eq(dozers[6]) & (cycles["Productivity (bcm/h)"] < 516)
]
cycles[cycles.Equipment_Code.eq(dozers[7])] = cycles[
    cycles.Equipment_Code.eq(dozers[7]) & (cycles["Productivity (bcm/h)"] < 450)
]

import plotly.express as px

fig = px.line(
    cycles.dropna(),
    x="Local_Datetime",
    y="Productivity (bcm/h)",
    color="Equipment_Code",
    color_discrete_sequence=color_list,
)
fig.show()


###########################################################
# Group into dozer groups DZ21 and DZ80
import plotly.graph_objects as go
from plotly.subplots import make_subplots

straight_cycles_copy = straight_cycles.copy()
straight_cycles_copy["dozerseries"] = straight_cycles_copy["equipment"].str[0:4]
straight_cycles_copy["datetime"] = positions.loc[straight_cycles_copy.idx_start]["datetime"].reset_index(drop=True)
straight_cycles_copy = straight_cycles_copy[straight_cycles_copy["total_cycle_time"] > 200]

# Grade
symbol_dict = color_dict = {
    "DZ21": "circle",
    "DZ80": "diamond",
}

dozers = straight_cycles_copy.dozerseries.unique()
color_dict = {
    "DZ21": "blue",
    "DZ80": "red",
}
straight_cycles_copy = straight_cycles_copy.set_index("datetime")
straight_cycles_copy = straight_cycles_copy.groupby(
    ["dozerseries", pd.Grouper(freq="60Min", base=30, label="right")]
).mean()
straight_cycles_copy = straight_cycles_copy.reset_index(names=["dozerseries", "datetime"])


fig = make_subplots(
    shared_xaxes=True,
    shared_yaxes=True,
    rows=1,
    cols=1,
)

# First Subplot
for dozer in dozers:
    fig.add_trace(
        go.Scatter(
            x=straight_cycles_copy[straight_cycles_copy.dozerseries.eq(dozer)].datetime,
            y=straight_cycles_copy[straight_cycles_copy.dozerseries.eq(dozer)].productivity,
            name=dozer,
            marker=dict(symbol=symbol_dict[dozer]),
            legendgroup=dozer,
            line=go.scatter.Line(color=color_dict[dozer]),
        ),
        row=1,
        col=1,
    )
fig.update_layout(
    title_text="Hourly Mean Productivity Per Dozer Series Across Shift",
    xaxis_title="Time of Push Cycle Aggregate",
)
fig.update_yaxes(title_text=f"Mean<br>Productivity (bcm/h)", row=1, col=1, secondary_y=False)

fig.update_layout(
    font=dict(size=19),
)
fig.update_traces(
    marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
)
fig.show()
