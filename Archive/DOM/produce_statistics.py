import pandas as pd
import pyodbc

from dozerpush.dozer_push import dozer_push
from dozerpush.dozer_acquire import acquire_shift_data

if __name__ == "__main__":
    Driver = "{ODBC Driver 17 for SQL Server}"
    Server = "tcp:tpl-psg-sql1.database.windows.net,1433"
    Database = "PAMSArchive"
    Encrypt = "yes"
    TrustServerCertificate = "no"
    ConnectionTimeout = "Connection Timeout = 60"
    Authentication = "ActiveDirectoryIntegrated"
    connection_string = (
        f"Driver={Driver};"
        f"Server={Server};"
        f"Database={Database};"
        f"Encrypt={Encrypt};"
        f"TrustServerCertificate={TrustServerCertificate};"
        f"{ConnectionTimeout};"
        f"Authentication={Authentication};"
    )
    cnxn = pyodbc.connect(connection_string)

    positions = acquire_shift_data(
        "VER", 9, "Night", "2023-02-09 18:30:00", "2023-02-10 06:30:00", "Australia/Brisbane", cnxn
    )
    # Add distances and durations
    # positions = pd.read_csv(".\\dozerpush\\test\\positions.csv")
    # positions.datetime = pd.to_datetime(positions.datetime)
    print("      Processing data...", flush=True)
    straight_cycles, positions = dozer_push(
        positions,
        window_distance_smoothing=30.0,
        grid_size=10.0,
        heading_deviation_max=20.0,
        push_speed_threshold=4.8,
        min_distance=30.0,
        max_distance=25.0,
        max_heading_delta=45.0,
        max_cluster_std=200.0,
    )

    ##########################################################################
    # TODO: WRITE THESE STATS TO FILE
    ##########################################################################
    positions.to_csv(".\\dozerpush\\test\\dozer_fms_stats_VER_2023-02-09_Night.csv", index=False)
    straight_cycles.to_csv(".\\dozerpush\\test\\dozer_summary_stats_VER_2023-02-09_Night.csv", index=False)
