"""Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
"""
from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from enum import Enum

# class LocationEnum(Enum):
#     SC_1 = 0
#     SC_1_TAKEIN = 1
#     SC_1_RETURN = 2
#     PICKUP_0 = 3
#     PICKUP_1 = 4
#     RETURN_0 = 5

class LocationEnum(Enum):
    SC_1 = 0
    SC_1_TAKEIN_0 = 1
    SC_1_TAKEIN_1 = 2
    SC_1_RETURN_1 = 3
    SC_2 = 4
    SC_2_TAKEIN_0 = 5
    SC_2_TAKEIN_1 = 6
    SC_2_RETURN_0 = 7
    V0_DEFAULT_START = 8
    V1_DEFAULT_START = 9
    V2_DEFAULT_START = 10
    V0_END = 11  # this variable is only used for setting the ending-shift time, not care about the location
    V1_END = 12
    V2_END = 13
    PICKUP_0 = 14
    PICKUP_1 = 15
    RETURN_0 = 16
    RETURN_1 = 17 # WEN TODO: problem is need to go back to depot in the middle of the route, how about refuel


###########################
# Problem Data Definition #
###########################
def create_data_model():
    """Stores the data for the problem"""
    data = {}
    # Locations in block units
    _locations = \
          [LocationEnum.SC_1,
           LocationEnum.SC_1_TAKEIN_0,
           LocationEnum.SC_1_TAKEIN_1,
           LocationEnum.SC_1_RETURN_1,
           LocationEnum.SC_2,
           LocationEnum.SC_2_TAKEIN_0,
           LocationEnum.SC_2_TAKEIN_1,
           LocationEnum.SC_2_RETURN_0,
           LocationEnum.V0_DEFAULT_START,
           LocationEnum.V1_DEFAULT_START,
           LocationEnum.V2_DEFAULT_START,
           LocationEnum.V0_END,
           LocationEnum.V1_END,
           LocationEnum.V2_END,
           LocationEnum.PICKUP_0,
           LocationEnum.PICKUP_1,
           LocationEnum.RETURN_0,
           LocationEnum.RETURN_1]
    # Multiply coordinates in block units by the dimensions of an average city block, 114m x 80m,
    # to get location coordinates.
    data["locations"] = [(0, 0) for l in range(len(_locations))]
    data["num_locations"] = len(data["locations"])
    data["num_vehicles"] = 3
    data["depot"] = 0

    #TODO: negative demand ok?
    demands = [0, -1, -1, 1,
               0, -1, -1, 1,
               0,0,0,
               0,0,0,
               1, 1,  -1, -1]

    #demands = [0, 0, 0, 0, 0, 0, 0,0 ,0,0,0,0]

    capacities = [1, 1, 1]

    # WEN NOTE: if a specialist has two shifts a day, treat it as 2 vehicles
    # TODO: Add End node for each vehicle (i.e. time off work)
    time_windows = \
            [(0, 160), (80, 100), (0,160), (80,100),
             (0,160), (0, 160), (0,160), (0,160),
             (35, 35), (50, 50), (40, 40), # Wen: speicalist starting working time
             (130, 130), (110, 110), (150, 150), # speicalist ending working time
                (55, 60), # 1, 2
                (90, 95),
                (80,85), (130,135)] # 15, 16

    # Multiply coordinates in block units by the dimensions of an average city block, 114m x 80m,
    # to get location coordinates.
    #data["locations"] = [(l[0] * 114, l[1] * 80) for l in _locations]

    data["demands"] = demands
    data["vehicle_capacities"] = capacities
    data["time_windows"] = time_windows
    data["time_per_demand_unit"] = 5
    data["vehicle_speed"] = 1
    return data
#######################
# Problem Constraints #
#######################
def manhattan_distance(position_1, position_2):
    """Computes the Manhattan distance between two points"""
    return (
        abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1]))
def create_distance_callback(data):
    """Creates callback to return distance between points."""
    _distances = {}


    for from_node in range(data["num_locations"]):
        _distances[from_node] = {}

    _distances[LocationEnum.SC_1.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_1_TAKEIN_0.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_1_TAKEIN_1.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_1_RETURN_1.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_2.value] = 35 #public transportation (fail at 65)
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_1.value][LocationEnum.SC_2_RETURN_0.value] = 35 #public transportation (fail at 65)
    _distances[LocationEnum.SC_1.value][LocationEnum.V0_DEFAULT_START.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.V1_DEFAULT_START.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.V2_DEFAULT_START.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.SC_1.value][LocationEnum.PICKUP_0.value] = 40
    _distances[LocationEnum.SC_1.value][LocationEnum.PICKUP_1.value] = 50
    _distances[LocationEnum.SC_1.value][LocationEnum.RETURN_0.value] = 30
    _distances[LocationEnum.SC_1.value][LocationEnum.RETURN_1.value] = 30

    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_1_TAKEIN_0.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_1_TAKEIN_1.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_1_RETURN_1.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_2.value] = 65 #public transportation
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.SC_2_RETURN_0.value] = 65 #public transportation
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_0.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_1_TAKEIN_0.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_1_TAKEIN_1.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_1_RETURN_1.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_2.value] = 65  # public transportation
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.SC_2_RETURN_0.value] = 65  # public transportation
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.SC_1_TAKEIN_1.value][LocationEnum.RETURN_1.value] = 1000

    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.SC_1.value] = 0
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.SC_1_TAKEIN_0.value] = 0
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.SC_1_TAKEIN_1.value] = 0
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.SC_1_RETURN_0.value] = 0
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.SC_1_RETURN_1.value] = 0
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.PICKUP_0.value] = 100
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.PICKUP_1.value] = 100
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.RETURN_0.value] = 35
    # _distances[LocationEnum.SC_1_RETURN_0.value][LocationEnum.RETURN_1.value] = 100

    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_1.value] = 0 #???
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_1_RETURN_1.value] = 0
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.SC_2_RETURN_0.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.SC_1_RETURN_1.value][LocationEnum.RETURN_1.value] = 10

#-------
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_1.value] = 70 #public transportation
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_1_RETURN_1.value] = 70 #public transportation
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_2.value] = 0
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_2_TAKEIN_0.value] = 0
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_2_TAKEIN_1.value] = 0
    _distances[LocationEnum.SC_2.value][LocationEnum.SC_2_RETURN_0.value] = 0
    _distances[LocationEnum.SC_2.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.PICKUP_0.value] = 40
    _distances[LocationEnum.SC_2.value][LocationEnum.PICKUP_1.value] = 50
    _distances[LocationEnum.SC_2.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.SC_2.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_1.value] = 70 #public transportation
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_1_RETURN_1.value] = 70 #public transportation
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_2.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_2_TAKEIN_0.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_2_TAKEIN_1.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.SC_2_RETURN_0.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_0.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_1.value] = 70 #public transportation
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_1_RETURN_1.value] = 70 #public transportation
                # we know the return 1 appt, so we know when we should start from sc_1, which becomes the arrival time for sc2 ->sc1
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_2.value] = 0  # public transportation
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_2_TAKEIN_0.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_2_TAKEIN_1.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.SC_2_RETURN_0.value] = 0  # public transportation
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.SC_2_TAKEIN_1.value][LocationEnum.RETURN_1.value] = 1000


    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_1.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.SC_2_RETURN_0.value] = 0
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.RETURN_0.value] = 25
    _distances[LocationEnum.SC_2_RETURN_0.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_1.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_1_RETURN_1.value] = 5
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.SC_2_RETURN_0.value] = 5
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.V0_DEFAULT_START.value] = 0
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.PICKUP_0.value] = 15
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.PICKUP_1.value] = 15
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.V0_DEFAULT_START.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_1_RETURN_1.value] = 5
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.SC_2_RETURN_0.value] = 5
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.V1_DEFAULT_START.value] = 0
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.PICKUP_0.value] = 15
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.PICKUP_1.value] = 15
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.V1_DEFAULT_START.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_1_RETURN_1.value] = 5
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.SC_2_RETURN_0.value] = 5
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.V2_DEFAULT_START.value] = 0
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.PICKUP_0.value] = 15
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.PICKUP_1.value] = 15
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.V2_DEFAULT_START.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.V0_END.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.SC_2_RETURN_0.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.V0_END.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.V0_END.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.V1_END.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.SC_2_RETURN_0.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.V1_END.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.V1_END.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.V2_END.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.SC_2_RETURN_0.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.V2_END.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.V2_END.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_1.value] = 0 # allow some vehicle to idle, not doing any job
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_1_TAKEIN_0.value] = 25
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_2_TAKEIN_0.value] = 35 # TODO: not optimal if 20
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_2_RETURN_0.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.PICKUP_0.value] = 0
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_1.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000 #different incidents have differetn travel time to the same SC
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_1_TAKEIN_1.value] = 5
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_2_TAKEIN_1.value] = 25
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_2_RETURN_0.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.V0_END.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.V1_END.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.V2_END.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.PICKUP_1.value] = 0
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.SC_2_RETURN_0.value] = 35
    _distances[LocationEnum.RETURN_0.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.RETURN_0.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.RETURN_0.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.RETURN_0.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.RETURN_0.value][LocationEnum.PICKUP_0.value] = 10
      # Here is the tricky part, the traffice time depends on the leaving hour,
     # he can wait at from or to locations. Maybe we can pre-compute the minimum time
    _distances[LocationEnum.RETURN_0.value][LocationEnum.PICKUP_1.value] = 5
    _distances[LocationEnum.RETURN_0.value][LocationEnum.RETURN_0.value] = 0
    _distances[LocationEnum.RETURN_0.value][LocationEnum.RETURN_1.value] = 1000

    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_1.value] = 0
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_1_TAKEIN_0.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_1_TAKEIN_1.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_1_RETURN_1.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_2.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_2_TAKEIN_0.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_2_TAKEIN_1.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.SC_2_RETURN_0.value] = 60 #tricky
    _distances[LocationEnum.RETURN_1.value][LocationEnum.V0_DEFAULT_START.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.V1_DEFAULT_START.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.V2_DEFAULT_START.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.V0_END.value] = 0
    _distances[LocationEnum.RETURN_1.value][LocationEnum.V1_END.value] = 0
    _distances[LocationEnum.RETURN_1.value][LocationEnum.V2_END.value] = 0
    _distances[LocationEnum.RETURN_1.value][LocationEnum.PICKUP_0.value] = 10 #tricky
    _distances[LocationEnum.RETURN_1.value][LocationEnum.PICKUP_1.value] = 5 #tricky
    _distances[LocationEnum.RETURN_1.value][LocationEnum.RETURN_0.value] = 1000
    _distances[LocationEnum.RETURN_1.value][LocationEnum.RETURN_1.value] = 0


        # for to_node in range(data["num_locations"]):
        #     if from_node == to_node:
        #         _distances[from_node][to_node] = 0
        #     else:
        #         _distances[from_node][to_node] = (
        #             manhattan_distance(data["locations"][from_node],
        #                        data["locations"][to_node]))

    def distance_callback(from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[from_node][to_node]

    return distance_callback

def create_demand_callback(data):
  """Creates callback to get demands at each location."""
  def demand_callback(from_node, to_node):
    return data["demands"][from_node]
  return demand_callback


def add_capacity_constraints(routing, data, demand_evaluator):
  """Adds capacity constraint"""
  capacity = "Capacity"
  routing.AddDimensionWithVehicleCapacity(
      demand_evaluator,
      0, # null capacity slack
      data["vehicle_capacities"], # vehicle maximum capacities
      True, # start cumul to zero
      capacity)

def create_time_callback(data):
  """Creates callback to get total times between locations."""
  def service_time(node):
    """Gets the service time for the specified location."""
    return data["demands"][node] * data["time_per_demand_unit"]

  def travel_time(from_node, to_node):
    """Gets the travel times between two locations."""
    if from_node == to_node:
      travel_time = 0
    else:
      travel_time = manhattan_distance(
                data["locations"][from_node],
                data["locations"][to_node]) / data["vehicle_speed"]
    return travel_time

  def time_callback(from_node, to_node):
    """Returns the total time between the two nodes"""
    serv_time = service_time(from_node)
    trav_time = travel_time(from_node, to_node)
    return serv_time + trav_time

  return time_callback
def add_time_window_constraints(routing, data, time_callback):
  """Add Global Span constraint"""
  time = "Time"
  horizon = 160
  routing.AddDimension(
    time_callback,
    horizon, # allow waiting time
    horizon, # maximum time per vehicle
    False, # Don't force start cumul to zero. This doesn't have any effect in this example,
           # since the depot has a start window of (0, 0).
    time)
  time_dimension = routing.GetDimensionOrDie(time)
  for location_node, location_time_window in enumerate(data["time_windows"]):
        index = routing.NodeToIndex(location_node)
        time_dimension.CumulVar(index).SetRange(location_time_window[0], location_time_window[1])

###########
# Printer #
###########
def print_solution(data, routing, assignment):
  """Prints assignment on console"""
  # Inspect solution.
  capacity_dimension = routing.GetDimensionOrDie('Capacity')
  time_dimension = routing.GetDimensionOrDie('Time')
  total_dist = 0
  time_matrix = 0

  for vehicle_id in range(data["num_vehicles"]):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
    route_dist = 0
    while not routing.IsEnd(index):
      node_index = routing.IndexToNode(index)

      next_node_index = routing.IndexToNode(
        assignment.Value(routing.NextVar(index)))
      route_dist += manhattan_distance(
        data["locations"][node_index],
        data["locations"][next_node_index])
      load_var = capacity_dimension.CumulVar(index)
      route_load = assignment.Value(load_var)
      time_var = time_dimension.CumulVar(index)
      time_min = assignment.Min(time_var)
      time_max = assignment.Max(time_var)

      if node_index > 0:
        plan_output += ' {0} Load({1}) Time({2},{3}) ->'.format(
            LocationEnum(node_index).name,
            route_load,
            time_min, time_max)
      index = assignment.Value(routing.NextVar(index))

    node_index = routing.IndexToNode(index)
    load_var = capacity_dimension.CumulVar(index)
    route_load = assignment.Value(load_var)
    time_var = time_dimension.CumulVar(index)
    route_time = assignment.Value(time_var)
    time_min = assignment.Min(time_var)
    time_max = assignment.Max(time_var)
    total_dist += route_dist
    time_matrix += route_time
    # skip last node (dummy depot)
    #plan_output += ' {0} Load({1}) Time({2},{3})\n'.format(LocationEnum(node_index).name, route_load,
    #                                                      time_min, time_max)
    plan_output += ' Off Work \n'
    plan_output += 'Distance of the route: {0} m\n'.format(route_dist)
    plan_output += 'Load of the route: {0}\n'.format(route_load)
    plan_output += 'Time of the route: {0} min\n'.format(route_time)
    print(plan_output)
  print('Total Distance of all routes: {0} m'.format(total_dist))
  print('Total Time of all routes: {0} min'.format(time_matrix))


########
# Main #
########
def main():
  """Entry point of the program"""
  # Instantiate the data problem.
  data = create_data_model()

  # Create Routing Model
  routing = pywrapcp.RoutingModel(data["num_locations"], data["num_vehicles"], data["depot"])
  # Define weight of each edge
  distance_callback = create_distance_callback(data)
  routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
  # Add Capacity constraint
  demand_callback = create_demand_callback(data)
  add_capacity_constraints(routing, data, demand_callback)
  # Add Time Window constraint
  time_callback = distance_callback #create_time_callback(data)
  add_time_window_constraints(routing, data, time_callback)

  # WEN TODO: What is it can deliver to SC_1 or SC_2 or SC_3
  #routing.AddPickupAndDelivery(LocationEnum.PICKUP_0.value, LocationEnum.SC_1_TAKEIN_0.value)

  v0_default_node_idx = routing.NodeToIndex(8)
  v0_end_node_idx = routing.NodeToIndex(11)
  routing.solver().Add(routing.VehicleVar(routing.Start(0)) == routing.VehicleVar(LocationEnum.V0_DEFAULT_START.value))
  #routing.solver().Add(routing.VehicleVar(routing.End(0)) == routing.VehicleVar(LocationEnum.V0_END.value))
  #routing.solver().Add(routing.VehicleVar(v0_end_node_idx) == routing.VehicleVar(routing.End(0)))

  v1_default_node_idx = routing.NodeToIndex(9)
  v1_end_node_idx = routing.NodeToIndex(12)
  routing.solver().Add(routing.VehicleVar(v1_default_node_idx) == routing.VehicleVar(routing.Start(1)))
  #routing.solver().Add(routing.VehicleVar(v1_end_node_idx) == routing.VehicleVar(routing.End(1)))

  v2_default_node_idx = routing.NodeToIndex(10)
  routing.solver().Add(routing.VehicleVar(v2_default_node_idx) == routing.VehicleVar(routing.Start(2)))

  # TODO: punish for miss one in the list? or punish for couldn't do any one in the list?
  routing.AddDisjunction([LocationEnum.SC_1_TAKEIN_0.value,LocationEnum.SC_2_TAKEIN_0.value], 100000)
  routing.AddDisjunction([LocationEnum.SC_1_TAKEIN_1.value,LocationEnum.SC_2_TAKEIN_1.value], 100000)
  routing.AddDisjunction([LocationEnum.RETURN_0.value], 100000)
  routing.AddDisjunction([LocationEnum.RETURN_1.value], 100000)
  routing.AddDisjunction([LocationEnum.PICKUP_0.value], 100000)
  routing.AddDisjunction([LocationEnum.PICKUP_1.value], 100000)
 # routing.AddDisjunction([LocationEnum.SC_1_TAKEIN_0.value], 10000)
  routing.AddDisjunction([LocationEnum.SC_2.value], 0) # we actually don't need SC_2

  # ??? not working?
  routing.AddPickupAndDelivery(LocationEnum.SC_2_RETURN_0.value, LocationEnum.RETURN_0.value)
  routing.AddPickupAndDelivery(LocationEnum.SC_1_RETURN_1.value, LocationEnum.RETURN_1.value)
  #routing.AddPickupAndDelivery(LocationEnum.V0_DEFAULT_START.value, LocationEnum.V0_END.value)
  #routing.AddPickupAndDelivery(LocationEnum.V2_DEFAULT_START.value, LocationEnum.V2_END.value)
  #routing.AddPickupAndDelivery(LocationEnum.SC_1_TAKEIN_0.value, LocationEnum.SC_1_RETURN_1.value)


  # v0_end_node_idx = routing.NodeToIndex(11)
  # routing.solver().Add(routing.VehicleVar(v0_end_node_idx) == routing.VehicleVar(routing.End(0)))
  #
  # v1_end_node_idx = routing.NodeToIndex(12)
  # routing.solver().Add(routing.VehicleVar(v1_end_node_idx) == routing.VehicleVar(routing.End(1)))
  #
  # v2_end_node_idx = routing.NodeToIndex(13)
  # routing.solver().Add(routing.VehicleVar(v2_end_node_idx) == routing.VehicleVar(routing.End(2)))


  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
  routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
#    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
#   routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED)
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  if assignment:
    printer = print_solution(data, routing, assignment)

if __name__ == '__main__':
  main()