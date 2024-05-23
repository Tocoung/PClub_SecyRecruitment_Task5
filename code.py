import pandas as pd
import networkx as nx
import heapq
import time
import os

# Load datasets
passengers = pd.read_csv('passengers.csv')
flights = pd.read_csv('flights.csv')
canceled_flights = pd.read_csv('canceled.csv')

# Identify affected passengers
affected_flight_ids = set(canceled_flights['Canceled'])
affected_passengers = passengers[passengers['FID'].isin(affected_flight_ids)]

# Filter out canceled flights from the flights dataset
available_flights = flights[~flights['FID'].isin(affected_flight_ids)]

# Initialize passenger counts for each flight
initial_passenger_counts = passengers[~passengers['FID'].isin(affected_flight_ids)]['FID'].value_counts().to_dict()

# Create a graph of available flights
G = nx.MultiDiGraph()

for _, flight in available_flights.iterrows():
    flight_id = flight['FID']
    initial_passengers = initial_passenger_counts.get(flight_id, 0)
    G.add_edge(flight['DEP'], flight['ARR'], 
               key=flight_id,
               departure_time=flight['DEP_TIME'], 
               arrival_time=flight['ARR_TIME'], 
               capacity=flight['CAPACITY'], 
               passengers=initial_passengers)  # Track the number of allocated passengers

# Helper function to find a path
def find_path(G, source, target, max_layovers=3):
    MIN_LAYOVER_TIME = 900  # Example layover time of 30 minutes
    queue = [(0, source, [], 0)]  # (current_time, node, path, layovers)
    seen = set()
    while queue:
        current_time, node, path, layovers = heapq.heappop(queue)
        if (node, tuple(path)) in seen:
            continue
        seen.add((node, tuple(path)))
        if node == target and layovers <= max_layovers:
            return path
        if layovers > max_layovers:
            continue
        for neighbor in G[node]:
            for key, data in G[node][neighbor].items():
                if data['passengers'] < data['capacity']:
                    if not path or (data['departure_time'] - path[-1][1] >= MIN_LAYOVER_TIME and data['departure_time'] > path[-1][1]):
                        heapq.heappush(queue, (data['arrival_time'], neighbor, path + [(key, data['arrival_time'])], layovers + 1))
    return None

# Reallocation process
start_time = time.time()
reallocated_passengers = []
total_layovers = 0
total_time_diff = 0

for _, passenger in affected_passengers.iterrows():
    original_flight = flights[flights['FID'] == passenger['FID']].iloc[0]
    path = find_path(G, original_flight['DEP'], original_flight['ARR'])
    if path:
        layovers = len(path) - 1
        total_layovers += layovers
        flights_path = [flight_id for flight_id, _ in path]
        reallocated_passengers.append((passenger['PID'], layovers + 1, flights_path))
        arrival_time_diff = abs(flights[flights['FID'] == flights_path[-1]].iloc[0]['ARR_TIME'] - original_flight['ARR_TIME'])
        total_time_diff += arrival_time_diff
        # Update flight capacities
        for flight_id in flights_path:
            for u, v, key, data in G.edges(data=True, keys=True):
                if key == flight_id:
                    G[u][v][key]['passengers'] += 1
    else:
        reallocated_passengers.append((passenger['PID'], 0, []))

end_time = time.time()
runtime = (end_time - start_time) * 1000  # in milliseconds

# Calculate statistics
num_affected = len(affected_passengers)
num_reallocated = sum(1 for p in reallocated_passengers if p[1] > 0)
avg_layovers = total_layovers / num_reallocated if num_reallocated > 0 else 0
avg_time_diff = total_time_diff / num_reallocated if num_reallocated > 0 else 0

# Save stats.csv
stats = pd.DataFrame({
    'affected_passengers': [num_affected],
    'successfully_reallocated': [num_reallocated],
    'avg_layovers': [avg_layovers],
    'avg_time_diff': [avg_time_diff],
    'runtime_ms': [runtime]
})
stats.to_csv('stats.csv', index=False)

# Save allot.csv to a writable directory
output_directory = os.path.expanduser('~')  # User's home directory
output_path = os.path.join(output_directory, 'allot.csv')

with open(output_path, 'w') as f:
    for passenger_id, num_flights, flights in reallocated_passengers:
        f.write(f"{passenger_id},{num_flights}," + ",".join(map(str, flights)) + "\n")

print(f"Process completed successfully. Output files saved in {output_directory}.")
