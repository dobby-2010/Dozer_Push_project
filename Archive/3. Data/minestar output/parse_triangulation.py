# Import dependencies
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm


# Function to parse filetype into dataframe


def parse_file(file_path: str) -> pd.DataFrame:
    """
    Parses a file into a dataframe. The file contains a head, a footer and a data section. Before
    the data block there appears to be two numbers that denote the number of faces and the number
    of points. Then the data block contains the data for each face and the set of points for each.

    Parameters
    ----------
    file_path : str
        Path to the file to be parsed.

    Returns
    -------
    df : pandas.DataFrame
        Parsed dataframe.
    """

    # Read the file
    with open(file_path, mode="rb") as file:  # b is important -> binary
        byteData = file.read()

    # File Position
    N = len(byteData)  # length of the file
    posn = 0  # position in the file

    # === Header =============================================================
    # The file contains a header and a footer of sorts.
    # Starts with the header: "Created External"
    if byteData[0:16].decode("utf-8") == "Created External":
        posn = 16
        # Move the position along to the two numbers denoting the number of faces and the number of points
        # Find the break in header and values (which is a set of nulls)
        posn = byteData.find(b"Created External", posn)  # get position of break after end of header
        while byteData[posn] != 0:
            posn += 1
            if posn == N:
                break
        while byteData[posn] == 0:
            posn += 1
            if posn == N:
                break

        # We are now at the position of the first number after the break
        # Get all bytes after this point that are not 0 and convert them to integers
        point_num = int.from_bytes(
            byteData[posn : byteData.find(b"\x00", posn)], byteorder="big"
        )  # number of faces/points
        posn = byteData.find(b"\x00", posn)

        # move on to get the number of points
        while byteData[posn] != 0:
            posn += 1
            if posn == N:
                break
        while byteData[posn] == 0:
            posn += 1
            if posn == N:
                break

        face_num = int.from_bytes(
            byteData[posn : byteData.find(b"\x00", posn)], byteorder="big"
        )  # number of points/faces
        posn = byteData.find(b"\x00", posn)

        # the next position is the start of the data block
        # move on to get the number of points
        while byteData[posn] != 0:
            posn += 1
            if posn == N:
                break
        while byteData[posn] == 0:
            posn += 1
            if posn == N:
                break

    # === Points Data Block ==================================================
    # for each point add this to a point dataframe
    point_data = []
    while posn < N:
        # keep running through the rows until there is a break that indicates the start of the faces data block
        # There should be poin_num rows in the data block with the easting, nothing and elevation.
        numBytes = 24  # each row contains 24 bytes
        # Current value of posn is the start of the data block
        # get the values of the string at posn to posn+numBytes
        ind_poin_data = np.frombuffer(byteData[posn : (posn + numBytes)], ">f8")
        posn += numBytes

        # add ind_poin_data to point_data
        point_data.append(ind_poin_data)

        # temporary break condition
        if (posn == N) or (len(point_data) == point_num):
            break

    # convert point_data list into dataframe with columns easting, northing, elevation
    point_data = pd.DataFrame(point_data, columns=["easting", "northing", "elevation"])

    # check if the number of rows in the dataframe is equal to point_num
    try:
        assert point_data.shape[0] == point_num
    except:
        raise Exception("Data Mismatch")

    # convert point_data into a geodataframe by making a geometry column
    point_data["geometry"] = point_data.apply(
        lambda row: Point(row["easting"], row["northing"], row["elevation"]), axis=1
    )

    # === Faces Data Block ==================================================

    # The remainder of the file is the triangulation element. This section contains the data for each face/triangle made in triangulation.
    # The data is constructed in such a way that every 24 bits contain 3 numbers. These three numbers are the
    # index of the points in the point_data dataframe.

    # E.g.
    # 00 00 ee 8d 00 00 ee 97  00 00 ee 96 00 00 00 00  |................|
    # 00 00 00 00 00 00 00 00                           |....|

    # Therefore this triangle is composed of points 61069, 61079, 61078 from point_data df.
    # Therefore, shape constructed between point_data.iloc[61069], point_data.iloc[61079], point_data.iloc[61078]

    # Need to figure out when this starts/occurs and collect the next 24 bits of the data to get the three numbers until the footer of the file is reached.
    # Create a geodataframe from the point_data dataframe and the three points (making triangles) to get df of triangles.

    # collect series of 3 points starting at current positions because these start immediately after last point.
    face_data = []
    while posn < N:
        numBytes = 24  # each set of 24 bytes contains 3 numbers

        # break bit_strip into 3 numbers
        ind_face_data = np.frombuffer(byteData[posn : (posn + numBytes)], ">i4")

        # keep non-zero values in array
        ind_face_data = ind_face_data[ind_face_data != 0]

        # add ind_face_data to face_data
        face_data.append(ind_face_data)

        posn += numBytes  # move along to next set of 24 bytes

        if (posn == N) or (len(face_data) == face_num):
            break

    # convert face_data into dataframe with columns point_1, point_2, point_3
    face_data = pd.DataFrame(face_data, columns=["point_1", "point_2", "point_3"])

    # check if the number of rows in the dataframe is equal to face_num
    try:
        assert face_data.shape[0] == face_num
    except:
        raise Exception("Data Mismatch")

    # convert face_data into a geodataframe by making a polygon geometry column
    face_data["geometry"] = np.nan
    for i in tqdm(range(len(face_data))):
        face_data.loc[i, "geometry"] = Polygon(
            (
                (
                    point_data.iloc[face_data.loc[i, "point_1"] - 1]["easting"],
                    point_data.iloc[face_data.loc[i, "point_1"] - 1]["northing"],
                    point_data.iloc[face_data.loc[i, "point_1"] - 1]["elevation"],
                ),
                (
                    point_data.iloc[face_data.loc[i, "point_2"] - 1]["easting"],
                    point_data.iloc[face_data.loc[i, "point_2"] - 1]["northing"],
                    point_data.iloc[face_data.loc[i, "point_2"] - 1]["elevation"],
                ),
                (
                    point_data.iloc[face_data.loc[i, "point_3"] - 1]["easting"],
                    point_data.iloc[face_data.loc[i, "point_3"] - 1]["northing"],
                    point_data.iloc[face_data.loc[i, "point_3"] - 1]["elevation"],
                ),
            )
        )

    return point_data, face_data
