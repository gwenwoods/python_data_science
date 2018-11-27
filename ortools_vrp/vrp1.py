"""Vehicle Routing Problem"""
from __future__ import print_function
from test_vrp.constraint_solver import pywrapcp
from test_vrp.constraint_solver import routing_enums_pb2

###########################
# Problem Data Definition #
###########################

def create_data():
  """Stores the data for the problem"""
  # Locations
  num_vehicles = 4
  depot = [i for i in range(0,75)]
  # locations = \
  #               [(4, 4), # depot
  #                (2, 0), (8, 0), # row 0
  #                (0, 1), (1, 1),
  #                (5, 2), (7, 2),
  #                (3, 3), (6, 3),
  #                (5, 5), (8, 5),
  #                (1, 6), (2, 6),
  #                (3, 7), (6, 7),
  #                (0, 8), (7, 8)]
  #A = ['A'+ str(i) for i in range(0,25)]
  #B = ['B'+ str(i) for i in range(0,25)]
  #C = ['C'+ str(i) for i in range(0,25)]

  A = [(0,i)  for i in range(0,25)]
  B = [(1, i) for i in range(0, 25)]
  C = [(2, i) for i in range(0, 25)]

  locations = []
  locations.extend(A)
  locations.extend(B)
  locations.extend(C)
  locations.append('X1')
  locations.append('X2')
  locations.append('X3')
  locations.append('X4')
  locations.append('X5')
  locations.append('Y1')
  # X1 = None
  # X2 = None
  # X3 = None
  # X4 = None
  # X5 = None
  # Y1 = None

  print(locations[0])

  num_locations = len(locations)
  dist_matrix = {}

  for i in range(0,24):
    x = A[i];
    dist_matrix[x][A[24]] = 0
    dist_matrix[A[24]][A[i]] = 0
    dist_matrix[B[i]][B[24]] = 0
    dist_matrix[B[24]][B[i]] = 0
    dist_matrix[C[i]][C[24]] = 0
    dist_matrix[C[24]][C[i]] = 0
    dist_matrix[A[8]][X1] = 40

  for from_node in range(num_locations):
    dist_matrix[from_node] = {}

    for to_node in range(num_locations):
      dist_matrix[from_node][to_node] = (
        manhattan_distance(locations[from_node], locations[to_node]))

  return [num_vehicles, depot, locations, dist_matrix]

###################################
# Distance callback and dimension #
####################################

def manhattan_distance(position_1, position_2):
  """Computes the Manhattan distance between two points"""
  return (abs(position_1[0] - position_2[0]) +
          abs(position_1[1] - position_2[1]))

def CreateDistanceCallback(dist_matrix):

  def dist_callback(from_node, to_node):
    return dist_matrix[from_node][to_node]

  return dist_callback


def add_distance_dimension(routing, dist_callback):
  """Add Global Span constraint"""
  distance = "Distance"
  maximum_distance = 3000
  routing.AddDimension(
    dist_callback,
    0, # null slack
    maximum_distance, # maximum distance per vehicle
    True, # start cumul to zero
    distance)
  distance_dimension = routing.GetDimensionOrDie(distance)
  # Try to minimize the max distance among vehicles.
  distance_dimension.SetGlobalSpanCostCoefficient(100)

####################
# Get Routes Array #
####################
def get_routes_array(assignment, num_vehicles, routing):
  # Get the routes for an assignent and return as a list of lists.
  routes = []
  for route_nbr in range(num_vehicles):
    node = routing.Start(route_nbr)
    route = []

    while not routing.IsEnd(node):
      index = routing.NodeToIndex(node)
      route.append(index)
      node = assignment.Value(routing.NextVar(node))
    routes.append(route)
  return routes

########
# Main #
########

def main():
  """Entry point of the program"""
  # Instantiate the data problem.
  [num_vehicles, depot, locations, dist_matrix] = create_data()
  num_locations = len(locations)
  # Create Routing Model
  routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
  # Define weight of each edge
  dist_callback = CreateDistanceCallback(dist_matrix)
  routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
  add_distance_dimension(routing, dist_callback)
  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  routes = get_routes_array(assignment, num_vehicles, routing)
  print("Routes array:")
  print(routes)

if __name__ == '__main__':
  main()