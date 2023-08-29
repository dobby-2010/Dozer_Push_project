"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module queries the data from the database or acquires the data.

-------------------------------------------------------------------------------
"""
from datetime import datetime
from pyodbc import Connection
from pyproj import Transformer
import pandas as pd
import pytz
from dozerpush.dozer_timeline import filter_dozer_timeline

###############################################################################
# ACQUIRE DATA
##############################################################################
def acquire_shift_data(
    site: str,
    date: int,
    shift_desc: str,
    shift_start: str,
    shift_end: str,
    timezone: str,
    cnxn_pams: Connection,
) -> pd.DataFrame:

    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This function queries both Infomine and PAMSArchive databases to get the positions of the
        dozers during their primary working time (PWT).

        Infomine query retrieves PWT element of the time usage model (TUM) for dozers in a given
        shift.

        After converting the timestamps of these PWT elements to the local timezone, the positions
        of dozers along with their speed, heading and track between these PWT periods are queried
        from PAMSArchive.

        If there are any positions for the dozers in the shift they are transformed to the local
        coordinate system (ENU).
    ---------------------------------------------------------------------------

        Usage: acquire_shift_data(site, shift, cnxn_PAMS, cnxn_InfoMine)


        Inputs:         site            String          The site code to acquire dozer push data
                                                        for.


                        shift                           The shift information.


                        cnxn_PAMS       Connection      The pyodbc connection string required to
                                                        access the SQL database for PAMS.


                        cnxn_InfoMine   Connection      The pyodbc connection string required to
                                                        access the SQL database for Infomine.


        Outputs:        positions       DataFrame       The dataframe of positions in ENU, the
                                                        speed, track, heading, etc.

    """

    # Obtain server info
    # serverID = shifts.infomine_serverID_and_TimeZone_from_site_code(site, cnxn_InfoMine)
    # timezone = serverID["TimeZoneName"]
    # serverID = serverID["ServerID"]

    # Obtain Dozer PWT hours
    # query = """
    #     SELECT
    #         Equipment.Code AS Equipment
    #         ,Activities.Description AS Activity
    #         ,Reasons.Description AS Reason
    #         ,TUM.StartTime
    #         ,TUM.EndTime
    #         ,ServerID
    #         ,TUM.DataSourceID
    #     FROM
    #         [Infomine].[Production].[EquipmentActivities] AS TUM
    #     JOIN
    #         [Infomine].[Config].[Activities] AS Activities
    #         ON Activities.ActivityID = TUM.ActivityID
    #     JOIN
    #         [Infomine].[Config].[Reasons] AS Reasons
    #         ON Reasons.ReasonID = TUM.ReasonID
    #     JOIN
    #         [Infomine].[Config].[Equipment] AS Equipment
    #         ON Equipment.EquipmentID = TUM.ParentEquipmentID
    #     WHERE
    #         EndTime > CAST('%s' AS datetimeoffset)
    #         AND EndTime < CAST('%s' AS datetimeoffset)
    #         AND StartTime > CAST('%s' AS datetimeoffset)
    #         AND StartTime < CAST('%s' AS datetimeoffset)
    #         AND Equipment.Code LIKE 'DZ%%'
    #         AND ServerID = %d
    # """ % (
    #     shift.StartTime.isoformat(),
    #     (shift.EndTime + pd.Timedelta("12h")).isoformat(),
    #     (shift.StartTime + pd.Timedelta("-12h")).isoformat(),
    #     shift.EndTime.isoformat(),
    #     serverID,
    # )
    # dozer_push_times = pd.read_sql(query, cnxn_InfoMine)

    # For testing. Uncomment all commented sections and comment out this chunk
    dozer_push_times = pd.read_csv(f".\\dozerpush\\test\\{site}_{date}_{shift_desc}.csv")
    dozer_push_times.StartTime = pd.to_datetime(
        dozer_push_times.StartTime
    ).dt.to_pydatetime()  # .dt.tz_localize("utc").dt.tz_convert(timezone)

    dozer_push_times.EndTime = pd.to_datetime(
        dozer_push_times.EndTime
    ).dt.to_pydatetime()  # .dt.tz_localize("utc").dt.tz_convert(timezone)

    dozer_push_times = filter_dozer_timeline(dozer_push_times)

    # Any doer push?
    if dozer_push_times.shape[0] == 0:
        return pd.DataFrame([])

    dozer_push_times.loc[:, "StartTime"] = dozer_push_times.StartTime.apply(
        lambda x: max(
            pytz.timezone("Australia/Brisbane")
            .localize(datetime.strptime(shift_start, "%Y-%m-%d %H:%M:%S"))
            .astimezone(pytz.utc),
            x.tz_localize("utc"),
        ).astimezone(pytz.timezone("Australia/Brisbane"))
    )

    dozer_push_times.loc[:, "EndTime"] = dozer_push_times.EndTime.apply(
        lambda x: min(
            pytz.timezone("Australia/Brisbane")
            .localize(datetime.strptime(shift_end, "%Y-%m-%d %H:%M:%S"))
            .astimezone(pytz.utc),
            x.tz_localize("utc"),
        ).astimezone(pytz.timezone("Australia/Brisbane"))
    )

    # Uncomment this when done.
    # Clip times
    # dozer_push_times.loc[:, "StartTime"] = dozer_push_times.StartTime.apply(
    #     lambda x: max(shift.StartTime, x.tz_localize("utc")).tz_convert(timezone)
    # )
    # dozer_push_times.loc[:, "EndTime"] = dozer_push_times.EndTime.apply(
    #     lambda x: min(shift.EndTime, x.tz_localize("utc")).tz_convert(timezone)
    # )

    # Obtain GPS positions for each production dozing period
    queries = []
    for dozer_push in dozer_push_times.itertuples():

        # Acquire data
        query = """
            SELECT
                tP.datetime,
                tP.latitude,
                tP.longitude,
                tP.altitude,
                tP.track,
                tP.heading,
                tP.speed,
                tD.code as equipment,
                tS.site
            FROM
                PAMS.tPositions AS tP
            JOIN
                PAMS.sites AS tS
                ON tS.id = tP.siteid
            JOIN
                PAMS.tDevices AS tD
                ON tD.id = tP.deviceid
                AND tD.siteid = tP.siteid
            WHERE
                tP.datetime >= CAST('%s' AS datetimeoffset)
                AND tP.datetime <= CAST('%s' AS datetimeoffset)
                AND tS.site = '%s'
                AND tD.code = '%s'
        """ % (
            dozer_push.StartTime.isoformat(),
            dozer_push.EndTime.isoformat(),
            site,
            dozer_push.Equipment,
        )
        queries.append(query)
    query_list = [i + " UNION ALL " for i in queries]
    query_final = "".join(query_list)
    query_final = query_final[:-10]
    query_final = query_final.replace("\n", "")
    positions = pd.read_sql(query_final, cnxn_pams)

    # Any positions?
    if len(positions) == 0:
        return pd.DataFrame([])

    # Attach the location of the work to the positions
    # query = """
    #     select eq.Code  as EquipmentCode,
    #         eqp.ServerID,
    #         StartTime,
    #         EndTime,
    #         DataSourceID,
    #         loc.Code as LocationCode
    #     from Infomine.Production.EquipmentLocations as eqp
    #             join Infomine.Config.Locations as loc
    #                 on eqp.ServerID = loc.ServerID and eqp.LocationID = loc.LocationID and loc.ServerID = eqp.ServerID
    #             join Infomine.Config.Equipment as eq on eqp.ParentEquipmentID = eq.ParentEquipmentID
    #         and eqp.ServerID = %d
    #         and eqp.EndTime > CAST('%s' AS datetimeoffset)
    #         and eqp.EndTime < CAST('%s' AS datetimeoffset)
    #         and eqp.StartTime > CAST('%s' AS datetimeoffset)
    #         and eqp.StartTime < CAST('%s' AS datetimeoffset)
    #         and eq.code like 'DZ%'
    #     group by EquipmentLocationID,
    #             eqp.ServerID,
    #             eqp.LocationID,
    #             StartTime,
    #             EndTime,
    #             DataSourceID,
    #             eq.Code,
    #             loc.LocationID,
    #             loc.ServerID,
    #             loc.Code,
    #             loc.Description
    #     order by EquipmentCode
    # """ % (
    #     shift.StartTime.isoformat(),
    #     (shift.EndTime + pd.Timedelta("12h")).isoformat(),
    #     (shift.StartTime + pd.Timedelta("-12h")).isoformat(),
    #     shift.EndTime.isoformat(),
    #     serverID,
    # )
    # dozer_push_times = pd.read_sql(query, cnxn_InfoMine)
    dozer_push_locations = pd.read_csv(f".\\dozerpush\\test\\{site}_{date}_{shift_desc}_Locations.csv")
    dozer_push_locations = dozer_push_locations.sort_values(
        ["EquipmentCode", "StartTime", "EndTime", "DataSourceID"], ascending=True
    ).drop_duplicates(subset=["EquipmentCode", "StartTime", "EndTime"], keep="last")
    dozer_push_locations = dozer_push_locations.rename(
        columns={"LocationCode": "location", "EquipmentCode": "equipment"}
    )

    # Add location information to positions
    positions = positions.join(dozer_push_locations[["location", "equipment"]].set_index("equipment"), on=["equipment"])

    # Create pit dictionary
    pit_dict = {"FN": "FAR NORTH", "NO": "NORTH", "CO": "CENTRAL", "SO": "SOUTH", "FS": "FAR SOUTH"}

    # Split location into pit, strip, seam, block
    # Convention of location in InfoMINE is '_'
    positions["pit"] = pit_dict[positions["location"].iloc[0].split("_")[0]]
    positions["strip"] = positions["location"].iloc[0].split("_")[1]
    positions["block"] = int(positions["location"].iloc[0].split("_")[2][-2:])
    positions["seam"] = "VEM"
    positions = positions.drop("location", axis=1)

    # There may be duplicate points as result of query
    positions = positions.drop_duplicates()

    # Convert to dataframe
    positions = positions.sort_values(["equipment", "datetime"]).reset_index(drop=True)

    # Any positions?
    if positions.shape[0] == 0:
        return pd.DataFrame([])

    # Convert positions to eastings and northings
    trans = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{20355}",
        always_xy=True,
    )
    easting, northing = trans.transform(positions["longitude"].values, positions["latitude"].values)
    positions.loc[:, "easting"] = easting
    positions.loc[:, "northing"] = northing
    positions.loc[:, "elevation"] = positions["altitude"]
    positions.drop(columns=["latitude", "longitude", "altitude"], inplace=True)

    # Convert datetimes to site timezone and speeds to km/h
    positions.loc[:, "datetime"] = positions.datetime.dt.tz_localize("utc").dt.tz_convert(timezone)
    positions.loc[:, "speed"] *= 3.6

    positions.loc[:, "shift_date"] = shift_start[:10]
    positions.loc[:, "shift_description"] = shift_desc

    # Return data
    return positions
