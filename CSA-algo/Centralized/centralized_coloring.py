import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the floor plan dimensions
floor_width = 2000
floor_height = 1000

# Generate non-overlapping rooms with varying sizes
def generate_rooms():
    rooms = []
    room_min_size = 100
    room_max_size = 300
    np.random.seed(0)
    for _ in range(30):  # Adjust the number of rooms as needed
        while True:
            width = np.random.randint(room_min_size, room_max_size)
            height = np.random.randint(room_min_size, room_max_size)
            x = np.random.randint(0, floor_width - width)
            y = np.random.randint(0, floor_height - height)
            new_room = ((x, y), (x + width, y + height))
            if not any(overlap(new_room, existing_room) for existing_room in rooms):
                rooms.append(new_room)
                break
    return rooms

def overlap(room1, room2):
    (x1, y1), (x2, y2) = room1
    (x3, y3), (x4, y4) = room2
    return not (x1 >= x4 or x2 <= x3 or y1 >= y4 or y2 <= y3)

rooms = generate_rooms()

# Extract walls from rooms
def extract_walls(rooms):
    walls = []
    for room in rooms:
        (x1, y1), (x2, y2) = room
        walls.append(((x1, y1), (x1, y2)))  # Left wall
        walls.append(((x1, y2), (x2, y2)))  # Top wall
        walls.append(((x2, y2), (x2, y1)))  # Right wall
        walls.append(((x2, y1), (x1, y1)))  # Bottom wall
    return walls

walls = extract_walls(rooms)

# Define available bandwidths (in MHz) and corresponding preference weights
available_bandwidths = [20, 40, 80, 160]
bandwidth_weights = {20: 0.1, 40: 1, 80: 3, 160: 5}  # Strongly prefer 160 MHz

# Define available 5 GHz channels
available_channels = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 149, 153, 157, 161, 165]

# Place APs either within rooms or on walls and assign bandwidth and channel
def place_aps(rooms):
    aps = []
    for room in rooms:
        x1, y1 = room[0]
        x2, y2 = room[1]
        position = np.random.choice(['inside', 'wall'])
        if position == 'inside':
            x = np.random.uniform(x1 + 1, x2 - 1)
            y = np.random.uniform(y1 + 1, y2 - 1)
        else:
            wall = np.random.choice(['left', 'right', 'top', 'bottom'])
            if wall == 'left':
                x = x1
                y = np.random.uniform(y1, y2)
            elif wall == 'right':
                x = x2
                y = np.random.uniform(y1, y2)
            elif wall == 'top':
                x = np.random.uniform(x1, x2)
                y = y2
            elif wall == 'bottom':
                x = np.random.uniform(x1, x2)
                y = y1
        bandwidth = np.random.choice(available_bandwidths)
        channel = np.random.choice(available_channels)
        aps.append((x, y, bandwidth, channel))
    return aps

aps = place_aps(rooms)

# Draw the floor plan and AP locations
def draw_floor_plan():
    plt.figure(figsize=(20, 10))
    for room in rooms:
        plt.plot([room[0][0], room[1][0]], [room[0][1], room[0][1]], 'k-')
        plt.plot([room[1][0], room[1][0]], [room[0][1], room[1][1]], 'k-')
        plt.plot([room[1][0], room[0][0]], [room[1][1], room[1][1]], 'k-')
        plt.plot([room[0][0], room[0][0]], [room[1][1], room[0][1]], 'k-')
    for ap in aps:
        plt.plot(ap[0], ap[1], 'ro')
        plt.text(ap[0], ap[1], f"AP{aps.index(ap) + 1} ({ap[2]} MHz, Ch {ap[3]})", fontsize=10, ha='right')
    plt.xlim(0, floor_width)
    plt.ylim(0, floor_height)
    plt.title("Office Floor Plan with AP Locations, Bandwidths, and Channels")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(True)
    plt.show()

draw_floor_plan()

# Path loss model considering bandwidth
def path_loss(distance, walls, bandwidth):
    if distance == 0:
        return 0
    loss = 20 * np.log10(distance) + 10 * np.log10(bandwidth / 20)  # Adjust loss based on bandwidth
    wall_loss = 5  # dB per wall (example value)
    total_loss = loss + walls * wall_loss
    return total_loss

# Generate the topology graph with AP positions, channels, and bandwidths
def generate_topology_graph(aps, path_loss_threshold=100, rssi_threshold=-80):
    G = nx.Graph()
    positions = {i: (ap[0], ap[1]) for i, ap in enumerate(aps)}
    for i in range(len(aps)):
        G.add_node(i, pos=positions[i], channel=aps[i][3], bandwidth=aps[i][2])
    for i in range(len(aps)):
        for j in range(i + 1, len(aps)):
            distance = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            num_walls = sum(1 for wall in walls if line_intersects_rect(positions[i], positions[j], wall))
            interference = path_loss(distance, num_walls, min(aps[i][2], aps[j][2]))
            if np.any(interference < path_loss_threshold):  # Use np.any to handle array comparison
                color = 'red' if np.any(interference > rssi_threshold) else 'black'
                G.add_edge(i, j, weight=interference, color=color)
    return G


def line_intersects_rect(p1, p2, rect):
    """ Check if the line segment between p1 and p2 intersects the rectangle rect """
    x1, y1 = p1
    x2, y2 = p2
    (rx1, ry1), (rx2, ry2) = rect
    return (
        line_intersects_line((x1, y1), (x2, y2), (rx1, ry1), (rx1, ry2)) or
        line_intersects_line((x1, y1), (x2, y2), (rx1, ry2), (rx2, ry2)) or
        line_intersects_line((x1, y1), (x2, y2), (rx2, ry2), (rx2, ry1)) or
        line_intersects_line((x1, y1), (x2, y2), (rx2, ry1), (rx1, ry1))
    )

def line_intersects_line(p1, p2, p3, p4):
    """ Check if line segment p1-p2 intersects with line segment p3-p4 """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

# Draw the topology graph with node colors based on channels and include BW in labels
def draw_topology_graph(G, cqi_dict, title):
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[i]['channel'] for i in G.nodes]
    labels = {i: f"AP{i+1} (Ch {G.nodes[i]['channel']}, BW {G.nodes[i]['bandwidth']} MHz)" for i in G.nodes}
    
    # Determine edge colors based on channel overlap
    edge_colors = []
    for (u, v) in G.edges():
        if G.nodes[u]['channel'] == G.nodes[v]['channel']:
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color=node_colors, cmap=plt.cm.jet, edge_color=edge_colors)
    
    # Draw CQI on each node (formatted to two decimal places)
    for i in G.nodes:
        plt.text(pos[i][0], pos[i][1] - 20, f"CQI: {cqi_dict[i]:.2f}", fontsize=10, ha='center', color='blue')
    
    # Draw edge labels for RSSI (formatted to one decimal place)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    formatted_edge_labels = {edge: f"{weight:.1f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels)
    
    plt.title(title)
    plt.show()

# Utility Calculation considering RSSI, Noise, Airtime, BW, and Interference
def calculate_utility(aps, traffic_patterns, G):
    utility = 0
    cqi_dict = {}
    for i, ap in enumerate(aps):
        rssi = calculate_rssi(ap, aps)
        noise = calculate_noise(ap, aps)
        airtime = calculate_airtime(ap, traffic_patterns[i])
        bandwidth_preference = bandwidth_weights[ap[2]]  # Strongly prefer wider bandwidths
        interference = 0
        
        # Add a strong penalty if a 20 MHz bandwidth is used
        if ap[2] == 20:
            utility -= 100  # High penalty value to discourage 20 MHz selection
        if ap[2] == 40:
            utility -= 50  # High penalty value to discourage 20 MHz selection

        # Penalty for interference (same channel) with neighboring APs
        neighbors = list(G.neighbors(i))
        for neighbor in neighbors:
            if aps[neighbor][3] == ap[3]:  # Same channel as a neighboring AP
                interference += 200  # Penalty for using the same channel as a neighbor

        cqi = (rssi - noise) / (airtime + interference)  # Simplified CQI calculation
        cqi_dict[i] = cqi
        
        utility += bandwidth_preference * cqi

    return utility, cqi_dict

def calculate_rssi(ap, aps):
    # RSSI is calculated using a simple path loss model
    rssi = -path_loss(1, 0, ap[2])  # Assume 1m distance for self-RSSI
    return rssi

def calculate_noise(ap, aps):
    # Noise is the cumulative interference from all other APs in the network
    noise = 0
    for other_ap in aps:
        if other_ap != ap:
            distance = np.linalg.norm(np.array([ap[0], ap[1]]) - np.array([other_ap[0], other_ap[1]]))
            if distance > 0:
                num_walls = sum(1 for wall in walls if line_intersects_rect((ap[0], ap[1]), (other_ap[0], other_ap[1]), wall))
                noise += path_loss(distance, num_walls, other_ap[2])
    return noise

def calculate_airtime(ap, traffic):
    # Airtime is inversely proportional to the bandwidth and directly proportional to traffic load
    bandwidth = ap[2]
    airtime = traffic.sum() / bandwidth
    return airtime

# Simulate traffic patterns using the On-Off model
def simulate_traffic_patterns():
    num_AP = len(aps)
    sim_time = 100  # Length of the simulation time
    traffic = np.zeros((num_AP, sim_time))
    
    # Define parameters for the On-Off model
    lambda_on_AP = np.random.uniform(0.1, 0.5, num_AP)  # Rate for "on" periods
    lambda_off_AP = np.random.uniform(0.1, 0.5, num_AP)  # Rate for "off" periods
    max_load_AP = np.random.uniform(5, 10, num_AP)  # Maximum load when "on"

    for AP in range(num_AP):
        on_off = np.zeros(sim_time)
        state = 0  # Initial state is "off"
        index = 0
        while index < sim_time:
            if state == 0:
                time = np.random.exponential(1 / lambda_off_AP[AP])
                state = 1 * np.random.rand() * max_load_AP[AP]
            else:
                time = np.random.exponential(1 / lambda_on_AP[AP])
                state = 0
            num_steps = int(np.ceil(time))
            on_off[index:min(index + num_steps, sim_time)] = state
            index += num_steps
        traffic[AP, :] = on_off
    
    return traffic

# Plot traffic patterns to visualize the On-Off model
def plot_traffic_patterns(traffic_patterns):
    plt.figure(figsize=(20, 10))
    for i in range(traffic_patterns.shape[0]):
        plt.plot(range(traffic_patterns.shape[1]), traffic_patterns[i], label=f"AP{i+1}")
    plt.xlabel("Time")
    plt.ylabel("Traffic Load")
    plt.title("Traffic Patterns Over Time (On-Off Model)")
    plt.legend()
    plt.show()

# Simulated Annealing Algorithm for Optimization
def simulated_annealing(aps, traffic_patterns, initial_temperature, cooling_rate, max_iterations):
    current_solution = aps.copy()
    G = generate_topology_graph(current_solution)
    current_utility, current_cqi_dict = calculate_utility(current_solution, traffic_patterns, G)
    best_solution = current_solution
    best_utility = current_utility
    best_cqi_dict = current_cqi_dict
    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate a neighboring solution
        new_solution = generate_neighbor_solution(current_solution)
        G_new = generate_topology_graph(new_solution)
        new_utility, new_cqi_dict = calculate_utility(new_solution, traffic_patterns, G_new)

        # Print the current state
        print(f"Iteration {iteration+1}:")
        print(f"Current Utility: {current_utility}")
        print("AP Channels and Bandwidths:", [(ap[3], ap[2]) for ap in current_solution])
        
        # Acceptance probability
        if new_utility > current_utility or np.random.rand() < np.exp((new_utility - current_utility) / temperature):
            current_solution = new_solution
            current_utility = new_utility
            current_cqi_dict = new_cqi_dict

            if new_utility > best_utility:
                best_solution = new_solution
                best_utility = new_utility
                best_cqi_dict = new_cqi_dict

        # Cool down the temperature
        temperature *= cooling_rate

        if temperature < 1e-3:  # Termination condition
            print("Temperature is too low. Exiting the loop.")
            break

    return best_solution, best_utility, best_cqi_dict

def generate_neighbor_solution(aps):
    new_aps = aps.copy()
    ap_to_modify = np.random.randint(len(aps))
    new_bandwidth = np.random.choice(available_bandwidths, p=[0.05, 0.15, 0.3, 0.5])  # Strong bias toward wider channels
    new_aps[ap_to_modify] = (
        new_aps[ap_to_modify][0] + np.random.uniform(-10, 10),
        new_aps[ap_to_modify][1] + np.random.uniform(-10, 10),
        new_bandwidth,
        np.random.choice(available_channels)
    )
    return new_aps

# Example usage of Simulated Annealing
traffic_patterns = simulate_traffic_patterns()
plot_traffic_patterns(traffic_patterns)

# Run Simulated Annealing
initial_temperature = 10000
cooling_rate = 0.9999
max_iterations = 100000  # Adjust as needed
best_solution_sa, best_utility_sa, best_cqi_dict_sa = simulated_annealing(aps, traffic_patterns, initial_temperature, cooling_rate, max_iterations)

# Plot the best solution found by Simulated Annealing
G_best_sa = generate_topology_graph(best_solution_sa)
draw_topology_graph(G_best_sa, best_cqi_dict_sa, title="Best Solution - Simulated Annealing")

# Print the best utility found
print("Simulated Annealing Best Utility:", best_utility_sa)
