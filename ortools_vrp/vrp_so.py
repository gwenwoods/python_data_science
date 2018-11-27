from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

###########################
# Problem Data Definition #
###########################

from enum import Enum

class LocationEnum(Enum):
    SC_1_TAKEIN = 0
    SC_1_RETURN = 1
    PICKUP_0 = 2
    PICKUP_1 = 3

def create_data_model():
    """Stores the data for the problem"""
    data = {}
    # Locations in block units
    _locations = \
          [LocationEnum.SC_1_TAKEIN,
           LocationEnum.SC_1_RETURN,
           LocationEnum.PICKUP_0,
           LocationEnum.PICKUP_1]
    # Multiply coordinates in block units by the dimensions of an average city block, 114m x 80m,
    # to get location coordinates.
    data["locations"] = [(l, l) for l in range(len(_locations))]
    data["num_locations"] = len(data["locations"])
    data["num_vehicles"] = 2
    data["depot"] = 0
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

    _distances[0][0] = 0
    _distances[LocationEnum.SC_1_TAKEIN.value][LocationEnum.SC_1_RETURN.value] = 0
    _distances[LocationEnum.SC_1_TAKEIN.value][LocationEnum.PICKUP_0.value] = 40
    _distances[LocationEnum.SC_1_TAKEIN.value][LocationEnum.PICKUP_1.value] = 50
    _distances[LocationEnum.SC_1_RETURN.value][LocationEnum.SC_1_TAKEIN.value] = 0
    _distances[LocationEnum.SC_1_RETURN.value][LocationEnum.SC_1_RETURN.value] = 0
    _distances[LocationEnum.SC_1_RETURN.value][LocationEnum.PICKUP_0.value] = 40
    _distances[LocationEnum.SC_1_RETURN.value][LocationEnum.PICKUP_1.value] = 50
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_1_TAKEIN.value] = 30
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.SC_1_RETURN.value] = 1000
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.PICKUP_0.value] = 0
    _distances[LocationEnum.PICKUP_0.value][LocationEnum.PICKUP_1.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_1_TAKEIN.value] = 20
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.SC_1_RETURN.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.PICKUP_0.value] = 1000
    _distances[LocationEnum.PICKUP_1.value][LocationEnum.PICKUP_1.value] = 0

    #_distances[0][1] = 50
    #_distances[1][0] = 0
    #_distances[1][1] = 0
        # for to_node in range(data["num_locations"]):
        #   if from_node == to_node:
        #     _distances[from_node][to_node] = 0
        #   else:
        #     _distances[from_node][to_node] = (
        #         manhattan_distance(data["locations"][from_node],
        #                            data["locations"][to_node]))

    def distance_callback(from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[from_node][to_node]

    return distance_callback
def add_distance_dimension(routing, distance_callback):
    """Add Global Span constraint"""
    distance = 'Distance'
    maximum_distance = 3000  # Maximum distance per vehicle.
    routing.AddDimension(
        distance_callback,
        0,  # null slack
        maximum_distance,
        True,  # start cumul to zero
        distance)
    distance_dimension = routing.GetDimensionOrDie(distance)
    # Try to minimize the max distance among vehicles.
    distance_dimension.SetGlobalSpanCostCoefficient(100)
###########
# Printer #
###########
def print_solution(data, routing, assignment):
    """Print routes on console."""
    total_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(LocationEnum(routing.IndexToNode(index)).name)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        plan_output += ' {}\n'.format(LocationEnum(routing.IndexToNode(index)).name)
        plan_output += 'Distance of route: {}m\n'.format(distance)
        print(plan_output)
        total_distance += distance
    print('Total distance of all routes: {}m'.format(total_distance))
########
# Main #
########
def main():

    print (LocationEnum.PICKUP_0)
    """Entry point of the program"""
    # Instantiate the data problem.
    data = create_data_model()
    # Create Routing Model
    routing = pywrapcp.RoutingModel(
        data["num_locations"],
        data["num_vehicles"],
        data["depot"])
    # Define weight of each edge
    distance_callback = create_distance_callback(data)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
    add_distance_dimension(routing, distance_callback)
    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # pylint: disable=no-member
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        print_solution(data, routing, assignment)
if __name__ == '__main__':
  main()