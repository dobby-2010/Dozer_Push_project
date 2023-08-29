"""
Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module is to better categorise and define times of production dozing. This will ensure
    that stats aren't being generated for shifts where there was no production dozing.
-------------------------------------------------------------------------------
"""

import pandas as pd


#######################################################################################################################
def priority_plus_priority(dozer_timeline_1: pd.DataFrame, dozer_timeline_2: pd.DataFrame) -> pd.DataFrame:
    """
    Written by Dominic Cotter

    ---- Description ----------------------------------------------------------
    This function serves the purpose of combining two timelines that are of different priority.
    dozer_timeline_1 must have a lower priority than dozer_timeline_2.
    ---------------------------------------------------------------------------

        Usage: priority_plus_priority(dozer_timeline_1, dozer_timeline_2)


        Inputs:         dozer_timeline_1    DataFrame       InfoMine timeline activities for all
                                                            dozers over a shift with the lowest
                                                            priority.


                        dozer_timeline_2    DataFrame       InfoMine timeline activities for all
                                                            dozers over a shift with the highest
                                                            priority.


        Outputs:        ammended_times      DataFrame       Timeline of activities with appropriate
                                                            priority.
    """

    # dozer_timeline_2 takes priority
    # for every activity in the priority 2 timeline must be checked to see if there are any overlapping activities in priority 1 timeline
    # Approach is to rewrite dozer_timeline_1 and then add in the activities (concat) of dozer_timeline_2
    j_removal = []
    for i in range(len(dozer_timeline_2)):

        for j in range(len(dozer_timeline_1)):
            # Give me the list of activities in dozer_timeline_1 that overlap with the iterator's activity:

            ########## Case 1 ##############
            #         |------------|    timeline_1   +
            #                |---------------| timeline_2
            #                   =
            #         |--1---|------2--------|
            if (
                (dozer_timeline_2.iloc[i].StartTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].StartTime < dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime > dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime > dozer_timeline_1.iloc[j].StartTime)
            ):
                dozer_timeline_1.loc[j, "EndTime"] = dozer_timeline_2.iloc[i].StartTime

            ########## Case 2 ##############
            #                           |------------|    timeline_1   +
            #               |---------------| timeline_2
            #                   =
            #               |------2--------|----1---|
            elif (
                (dozer_timeline_2.iloc[i].StartTime < dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].StartTime < dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].EndTime < dozer_timeline_1.iloc[j].EndTime)
            ):
                dozer_timeline_1.loc[j, "StartTime"] = dozer_timeline_2.iloc[i].EndTime

            ########## Case 3 ##############
            #             |--------------------| timeline_1
            #                 |------------|    timeline_2   +
            #                   =
            #             |-1-|-----2------|-1-|
            elif (
                (dozer_timeline_2.iloc[i].StartTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].StartTime < dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].EndTime < dozer_timeline_1.iloc[j].EndTime)
            ):
                act_0 = dozer_timeline_1.iloc[j]  # capture timeline_1 details to create second activity from split
                act = act_0.copy()
                # Split timeline_1 and creates first activity
                dozer_timeline_1.loc[j, "EndTime"] = dozer_timeline_2.iloc[i].StartTime
                act.StartTime = dozer_timeline_2.iloc[i].EndTime
                dozer_timeline_1 = pd.concat(
                    [dozer_timeline_1, pd.DataFrame(act).T], axis=0
                )  # concat act onto dozer_timeline_1
                # This is a problem area because I am just dropping rows and replacing

            ########## Case 4 ##############
            #                   |--------| timeline_1
            #                 |------------|    timeline_2   +
            #                   =
            #                 |-----2------|
            elif (
                (dozer_timeline_2.iloc[i].StartTime <= dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].StartTime <= dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime >= dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].EndTime >= dozer_timeline_1.iloc[j].EndTime)
            ):
                # drop the activity from dozer-timeline_1 by adding row to list to be dropped
                j_removal.append(j)
                j_removal = list(set(j_removal))
                # dozer_timeline_1 = dozer_timeline_1.drop([j])

            ########## Case 5 ##############
            #                 |--------------------| timeline_1
            #                 |------------|    timeline_2   +
            #                   =
            #                 |-----2------|---1---|
            elif (
                (dozer_timeline_2.iloc[i].StartTime == dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].StartTime <= dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].EndTime < dozer_timeline_1.iloc[j].EndTime)
            ):
                # drop the activity from dozer-timeline_1 by adding row to list to be dropped
                dozer_timeline_1.loc[j, "StartTime"] = dozer_timeline_2.iloc[i].EndTime

            ########## Case 6 ##############
            #                 |--------------------| timeline_1
            #                         |------------|    timeline_2   +
            #                   =
            #                 |---1---|-----2------|
            elif (
                (dozer_timeline_2.iloc[i].StartTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].StartTime < dozer_timeline_1.iloc[j].EndTime)
                & (dozer_timeline_2.iloc[i].EndTime > dozer_timeline_1.iloc[j].StartTime)
                & (dozer_timeline_2.iloc[i].EndTime == dozer_timeline_1.iloc[j].EndTime)
            ):
                # drop the activity from dozer-timeline_1 by adding row to list to be dropped
                dozer_timeline_1.loc[j, "EndTime"] = dozer_timeline_2.iloc[i].StartTime

    dozer_timeline_1 = dozer_timeline_1.reset_index(drop=True).drop(j_removal)
    ammended_times = (
        pd.concat([dozer_timeline_1, dozer_timeline_2], axis=0).sort_values(["StartTime"]).reset_index(drop=True)
    )

    # Check if there are any activities that are the same and are directly next to each other:
    #         |---------||---------|
    #                   =
    #         |--------------------|
    removal_index_list = []
    for i in range(len(ammended_times) - 1):
        if (
            (ammended_times.iloc[i].Activity == ammended_times.iloc[i + 1].Activity)
            & (ammended_times.iloc[i].Reason == ammended_times.iloc[i + 1].Reason)
            & (ammended_times.iloc[i].ActivityCode == ammended_times.iloc[i + 1].ActivityCode)
        ):
            ammended_times.loc[i + 1, "StartTime"] = ammended_times.loc[i, "StartTime"]
            removal_index_list.append(i)

    ammended_times = ammended_times.drop(removal_index_list).reset_index(drop=True)
    return ammended_times


#######################################################################################################################
def filter_dozer_timeline(dozer_activities: pd.DataFrame) -> pd.DataFrame:
    """
        Written by Dominic Cotter

    ---- Description ----------------------------------------------------------
        This function incorporates the the priority_plus_priority to combine default, imported and
        specified timelines. This function, although labelled as filter_dozer_timeline, is capable
        of ordering and constructing the timelines of equipment (up until the specific step to
        filter the timeline for pdouction related activities). This function is based on the work
        of Daniel Wilder who wrote the C# code for the resolve timeline function.
    ---------------------------------------------------------------------------
        Usage: filter_dozer_timeline(dozer_activities)


        Inputs:         dozer_activities    DataFrame       InfoMine timeline activities for all
                                                            dozers over a shift.


        Outputs:        production_dozing_activities    DataFrame       Timeline of production
                                                                        activities.

    """

    production_dozing_activities = pd.DataFrame([])
    for dozer in dozer_activities.Equipment.unique():

        dozer_times = dozer_activities[dozer_activities.Equipment.eq(dozer)]
        source = sorted(set(dozer_times.DataSourceID.unique()))
        if len(source) == 2:
            source.append(5)
        if len(source) == 1:
            source.append(4)
            source.append(5)
        source = sorted(source)

        # Use the priority 3 activities to build timeline and fill in rest with priority 2 an 1 activities?

        # Get default timeline
        dozer_times_1 = (
            dozer_times[dozer_times.DataSourceID.eq(source[0])].sort_values("StartTime").reset_index(drop=True)
        )
        dozer_times_2 = (
            dozer_times[dozer_times.DataSourceID.eq(source[1])].sort_values("StartTime").reset_index(drop=True)
        )
        dozer_times_3 = (
            dozer_times[dozer_times.DataSourceID.eq(source[2])].sort_values("StartTime").reset_index(drop=True)
        )

        # ammended_times is the desired output that then needs to be filtered for activity code '00' or has reason production dozing or unmaned-prod. doz.
        if dozer_times_1.empty & dozer_times_2.empty & dozer_times_3.empty:
            continue
        elif dozer_times_1.empty & dozer_times_2.empty:
            ammended_times = dozer_times_3
        elif dozer_times_1.empty & dozer_times_3.empty:
            ammended_times = dozer_times_2
        elif dozer_times_2.empty & dozer_times_3.empty:
            ammended_times = dozer_times_1
        elif dozer_times_3.empty:
            ammended_times = priority_plus_priority(dozer_times_1, dozer_times_2)
        elif dozer_times_2.empty:
            ammended_times = priority_plus_priority(dozer_times_1, dozer_times_3)
        elif dozer_times_1.empty:
            ammended_times = priority_plus_priority(dozer_times_2, dozer_times_3)
        else:
            ammended_times_0 = priority_plus_priority(dozer_times_1, dozer_times_2)
            ammended_times = priority_plus_priority(ammended_times_0, dozer_times_3)

        # filter the ammended_times df for only the production dozing activity/reason
        ammended_times = ammended_times[ammended_times.ActivityCode.eq(0)]
        if ammended_times.empty:
            continue

        # concatenate ammended_times to one large csv of all dozers for given shift
        production_dozing_activities = pd.concat([production_dozing_activities, ammended_times])

    return production_dozing_activities
