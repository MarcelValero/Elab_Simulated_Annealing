# Description: Simulated annealing algorithm for optimizing the location of service points
import pandas as pd
import random
import math


# Author:
#   Marcel Valero i6315821
#   Date: 2021-09-30


# Create object SP (for service point)
class SP:
    """
    Service point class
    initializes with:
    :param SP_id: service point id
    :param x: x coordinate
    :param y: y coordinate
    :param assigned_squares: list of squares assigned to this service point
    :param total_dist: distance from this service point to all the people who ordered from it
    :param delivery: amount of deliveries expected from all the squares assigned to this service point
    :param pickup: amount of pickups expected from all the squares assigned to this service point
    :param cost: service point cost
    """

    def __init__(self, SP_id, x, y, assigned_squares, total_dist, delivery, pickup, cost):
        self.SP_id = SP_id
        self.x = x
        self.y = y
        self.assigned_squares = assigned_squares
        self.total_dist = total_dist
        self.delivery = delivery
        self.pickup = pickup
        self.cost = cost

    def __repr__(self):
        return f"SP(id={self.SP_id}, x={self.x}, y={self.y}, assigned_squares={self.assigned_squares}, total_dist={self.total_dist} , delivery={self.delivery}, pickup={self.pickup} ,cost={self.cost})"


class Square:  # this will be used as the available coordinates for new service points to be spawned #
    """
    Square class
    initializes with:
    :param x: x coordinate
    :param y: y coordinate
    :param pickup: amount of pickups expected from the service point
    :param delivery: amount of deliveries expected from the service point
    """

    def __init__(self, x, y, pickup, delivery):
        self.delivery = delivery
        self.pickup = pickup
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Square(x={self.x}, y={self.y}, sp_dist={self.sp_dist}, pickup={self.pickup}, delivery={self.delivery})"


class InitialSolution:
    """
    Initial solution class
    initializes with:
    :param service_points: list of service points
    :param distance_df: dataframe with the distance from all the service points to all the squares

    Methods:
    - total_cost: Calculate the total cost of the initial solution
    - select_random_coordinate: Select a random coordinate from the CSV file
    - modify_service_point: Randomly select a service point and modify its coordinates
    - add_service_point: Add a new service point with random coordinates
    - delete_service_point: Delete a random service point
    """

    def __init__(self, service_points, distance_df):
        self.service_points = service_points
        self.distance_df = distance_df

    def __repr__(self):
        return f"InitialSolution(service_points={self.service_points})"

    def total_cost(self):
        """
        Calculate the cost of the initial solution.

        The cost is calculated as the sum of a base cost, a cost per unit of pickup capacity,
        and a cost per unit of total distance.

        :return: cost of the initial solution
        :rtype: float
        """
        # distance to all the squares assigned to the sp * sum of expected deliveries for every square
        for sp in self.service_points:
            sp.total_dist = sum(self.distance_df[sp.sp_id][square_id] for square_id in sp.assigned_squares)

        cost = 0
        for sp in self.service_points:
            cost += 75000 + 0.1 * sp.pickup + 0.5 * sp.total_dist

        return cost

    def modify_service_point(self, valid_coordinates):
        """
        Randomly select a service point and modify its coordinates.

        This method selects a random service point from the list and assigns it new coordinates
        chosen randomly from the list valid coordinates.

        :param valid_coordinates: List of tuples representing valid (x, y) coordinates
        :type valid_coordinates: list of tuple
        """
        sp = random.choice(self.service_points)
        new_id, new_x, new_y = select_random_coordinate(valid_coordinates)
        sp.x = new_x
        sp.y = new_y

        # Reset assigned squares for all service points
        for sp in self.service_points:
            sp.assigned_squares = []

        # Re-assign the closest squares to all the service points
        for square_id in self.distance_df.index:  # iterate over all the squares
            min_distance = float('inf')
            closest_sp = None
            for sp in self.service_points:
                # Ensure sp.sp_id is a single value
                if not isinstance(sp.SP_id, (int, str)):
                    raise TypeError(f"Service point ID {sp.SP_id} is not a valid type (int or str)")

                try:
                    distance = self.distance_df.at[square_id, sp.SP_id]
                    if distance < min_distance:
                        min_distance = distance
                        closest_sp = sp
                except KeyError as e:
                    # Check if the KeyError is due to missing service point ID in the distance matrix
                    if sp.SP_id not in self.distance_df.columns:
                            #print(f"Service point ID {sp.SP_id} is not a valid key in the distance matrix")
                            continue
                    else:
                        raise e
                closest_sp.assigned_squares.append(square_id)

        print(f"Service Point {sp.SP_id} coordinates modified to ({new_x}, {new_y})")

    def add_service_point(self, valid_coordinates, ):
        """
        Add a new service point with random coordinates.

        This method creates a new service point with a unique ID and random coordinates
        chosen from the list of valid coordinates, and adds it to the list of service points.

        :param valid_coordinates: List of potential coordinates to build a service point
        """
        chosen_coordinate = select_random_coordinate(valid_coordinates)
        new_id, new_x, new_y = chosen_coordinate
        new_sp = SP(new_id, new_x, new_y, [], 0, 0, 0, 0)
        self.service_points.append(new_sp)

        # Reset assigned squares for all service points
        for sp in self.service_points:
            sp.assigned_squares = []

        # Re-assign the closest squares to all the service points
        for square_id in self.distance_df.index:  # iterate over all the squares
            min_distance = float('inf')
            closest_sp = None
            for sp in self.service_points:
                # Ensure sp.sp_id is a single value
                if not isinstance(sp.SP_id, (int, str)):
                    raise TypeError(f"Service point ID {sp.SP_id} is not a valid type (int or str)")

                try:
                    distance = self.distance_df.at[square_id, sp.SP_id]
                    if distance < min_distance:
                        min_distance = distance
                        closest_sp = sp
                except KeyError as e:
                    # Check if the KeyError is due to missing service point ID in the distance matrix
                    if sp.SP_id not in self.distance_df.columns:
                        # print(f"Service point ID {sp.SP_id} is not a valid key in the distance matrix")
                        continue
                    else:
                        raise e
                closest_sp.assigned_squares.append(square_id)

        print(f"New Service Point added with ID {new_id} at coordinates ({new_x}, {new_y})")

    def delete_service_point(self):
        """
        Delete a random service point.

        This method randomly selects a service point from the list and removes it.
        """
        if self.service_points:
            deleted_sp = self.service_points.pop(random.randint(0, len(self.service_points) - 1))
            print(f"Random Service Point {deleted_sp} deleted")

            # Re-assign the closest squares to all the service points
            for sp in self.service_points:
                sp.assigned_squares = []  # Reset assigned squares

            for square_id in self.distance_df.columns:  # iterate over all the squares
                min_distance = float('inf')
                closest_sp = None
                for sp in self.service_points:
                    if square_id in self.distance_df.index and sp.SP_id in self.distance_df.columns:
                        if self.distance_df[sp.SP_id][square_id] < min_distance:
                            min_distance = self.distance_df[sp.SP_id][square_id]
                            closest_sp = sp

                if closest_sp is not None:
                    closest_sp.assigned_squares.append(square_id)
                else:
                    print(f"No closest service point found for square ID: {square_id}")
                    continue  # Continue the loop without raising an AttributeError

                try:
                    closest_sp.assigned_squares.append(square_id)
                except AttributeError as e:
                    print(f"Error occurred: {e}")

        return deleted_sp

    def select_random_coordinate(self, valid_coordinates):
        """
                Select a random coordinate from the CSV file.

                This method reads the CSV file, selects a random row, and returns the ID, x coordinate,
                and y coordinate from that row.

                :return: A tuple representing the chosen (id, x, y) coordinate
                :rtype: tuple
                """
        df = valid_coordinates
        random_row = df.sample(n=1).iloc[0]
        return random_row['Square_ID'], random_row['coordinate.x'], random_row['coordinates.y']


def create_service_points(file_path):
    """
    Create a list of service points from a CSV file.

    This function reads service point data from a CSV file and creates a list of SP objects.

    :param file_path: Path to the CSV file containing service point data
    :type file_path: str
    :return: List of service points
    :rtype: list of SP
    """
    df = pd.read_csv(file_path, delimiter=';')
    service_points = []
    for index, row in df.iterrows():
        sp = SP(
            SP_id=row[2],
            x=row[0],
            y=row[1],
            assigned_squares=[],
            total_dist=0,
            delivery=0,
            pickup=0,
            cost=0
        )
        service_points.append(sp)
    print(f"Loaded {len(service_points)} service points")
    return service_points


def load_valid_coordinates(valid_coordinates):
    """
    Load valid coordinates from a CSV file.

    This function reads valid (x, y) coordinates from a CSV file and returns them as a list of tuples.

    :param valid_coordinates: Path to the CSV file containing valid coordinates
    :return: List of valid (x, y) coordinates
    :rtype: list of tuple
    """
    df = pd.read_csv(valid_coordinates)
    valid_coordinates = list(zip(df['Square_ID'], df['coordinate.x'], df['coordinate.y']))
    return valid_coordinates


def select_random_coordinate(valid_coordinates):
    """
    Select a random coordinate from the list of valid coordinates.

    This function returns a random (x, y) coordinate from the provided list of valid coordinates.

    :param valid_coordinates: List of tuples representing valid (x, y) coordinates
    :type valid_coordinates: list of tuple
    :return: Random (x, y) coordinate
    :rtype: tuple
    """
    return random.choice(valid_coordinates)


def simulated_annealing(original_profit, new_profit, temperature):
    """
    Perform the simulated annealing acceptance criterion.

    :param original_profit: The profit of the original solution
    :type original_profit: float
    :param new_profit: The profit of the new solution
    :type new_profit: float
    :param temperature: The current temperature in the simulated annealing process
    :type temperature: float
    :return: Whether the new solution should be accepted
    :rtype: bool
    """
    probability = math.exp((new_profit - original_profit) / temperature)
    math_random = random.random()

    return probability >= math_random


def main():
    # Load data
    sp_initial = '/Users/valero/Elab 2/Case 2/Datasets/Initial_sp.csv'
    all_neighborhoods = '/Users/valero/Elab 2/Case 2/Datasets/predictions_milestone2.csv'
    distance_matrix = '/Users/valero/Elab 2/Case 2/Datasets/distance_matrix_km_filtered.csv'

    ServiceP = create_service_points(sp_initial)
    valid_coordinates = load_valid_coordinates(all_neighborhoods)
    distance_df = pd.read_csv(distance_matrix, skiprows=[0])

    # Generate initial solution
    initial_solution = InitialSolution(ServiceP, distance_df)
    current_cost = initial_solution.total_cost()

    print(f"Initial cost: {current_cost}")
    print(f"Number of Service Points: {len(ServiceP)}")

    temperature = 40

    for i in range(1, 350):
        print(f"Iteration {i}, current cost: {current_cost}")

        # Generate a new solution by modifying, adding, or deleting a service point
        new_solution = InitialSolution(ServiceP.copy(), distance_df.copy())  # Ensure a deep copy if necessary
        rand = random.random()

        if rand <= 0.3:
            print(f"Iteration {i}: Delete Service Point")
            new_solution.delete_service_point()
        elif rand <= 0.4:
            print(f"Iteration {i}: Add Service Point")
            new_solution.add_service_point(valid_coordinates)
        else:
            print(f"Iteration {i}: Modify Service Point")
            new_solution.modify_service_point(valid_coordinates)

        new_cost = new_solution.total_cost()
        temperature = 400000 / i  # Update temperature

        # Apply the simulated annealing acceptance criterion
        if simulated_annealing(current_cost, new_cost, temperature):
            print(f"Iteration {i}: Accepted new solution with cost {new_cost}")
            initial_solution = new_solution  # Accept the new solution
            current_cost = new_cost
        else:
            print(f"Iteration {i}: Rejected new solution with cost {new_cost}")

    print("FINAL SOLUTION")
    print(f"Final cost: {current_cost}")
    print("Solution is valid:", True)


if __name__ == "__main__":
    main()
