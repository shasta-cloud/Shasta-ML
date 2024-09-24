import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Set a random seed for reproducibility
np.random.seed(42)

def generate_neighbor_matrix(num_AP, density):
    neighbor_matrix = np.random.rand(num_AP, num_AP) < density
    np.fill_diagonal(neighbor_matrix, 0)  # AP cannot be its own neighbor
    return neighbor_matrix

def run_single_simulation(num_AP, sim_time, num_channels, scan_period, selection_periods, busy_factor, busy_factor_self, idle_factor, neighbor_matrix):
    lambda_on_AP = 0.1 * np.ones(num_AP)
    lambda_off_AP = 0.0001 * np.ones(num_AP)
    max_load_AP = np.random.rand(num_AP)
    start_sel = np.ceil(selection_periods * np.random.rand(num_AP)).astype(int)
    channels = np.ones(num_AP, dtype=int)
    scan_ch = np.ones(num_AP, dtype=int)
    channels_time = np.zeros((num_AP, sim_time))
    traffic = np.zeros((num_AP, sim_time))
    channel_switches = 0

    # Initialize RRM variables
    cahnnel_busy_AP = np.zeros((num_AP, num_channels))
    g_channel_busy = np.zeros((num_AP, num_channels))
    g_cur_channel_busy = np.zeros(num_AP)

    # Metric for distance from optimality
    distance_from_optimality = np.zeros(sim_time)

    for AP in range(num_AP):
        on_off = np.zeros(sim_time)
        state = 0  # Initial state is "off"
        index = 1
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

    # Main RRM loop
    for time in range(sim_time):
        for AP in range(num_AP):
            if time % scan_period == 0:
                # Channel scan
                ch = scan_ch[AP]
                if channels[AP] != ch:
                    cahnnel_busy_AP[AP, ch - 1] = min(np.sum((channels == ch) * traffic[:, time] * neighbor_matrix[AP, :]), 1)
                    if cahnnel_busy_AP[AP, ch - 1] > 0:
                        g_channel_busy[AP, ch - 1] = g_channel_busy[AP, ch - 1] * busy_factor + (1 - busy_factor) * cahnnel_busy_AP[AP, ch - 1]
                    else:
                        g_channel_busy[AP, ch - 1] = g_channel_busy[AP, ch - 1] * idle_factor
                scan_ch[AP] = (ch < num_channels) * (ch + 1) + (ch == num_channels) * 1
            if (time + start_sel[AP]) % selection_periods[AP] == 0:
                ch = channels[AP]
                busy = np.sum(np.mean((channels[:, np.newaxis] == ch) * traffic[:, max(0, time - selection_periods[AP]):time] * neighbor_matrix[AP, :, np.newaxis], axis=1))
                tx = np.mean(traffic[AP, max(0, time - selection_periods[AP]):time])
                busy -= tx
                g_cur_channel_busy[AP] = g_cur_channel_busy[AP] * busy_factor_self + (1 - busy_factor_self) * busy
                g_channel_busy[AP, ch - 1] = g_cur_channel_busy[AP]  # No penalty applied
                min_avg = np.min(g_channel_busy[AP, :])
                best_ch = np.argmin(g_channel_busy[AP, :]) + 1
                if g_cur_channel_busy[AP] > 1.25 * min_avg:# and g_cur_channel_busy[AP] > 0.15:
                    # Move channel
                    channels[AP] = best_ch
                    g_channel_busy[AP, :] = np.zeros(num_channels)
                    g_cur_channel_busy[AP] = 0
                    channel_switches += 1
            # Calculate distance from optimality
            current_ch = channels[AP] - 1
            if g_channel_busy[AP, current_ch] > np.min(g_channel_busy[AP, :]):
                distance_from_optimality[time] += 1
        channels_time[:, time] = channels

    return channel_switches, distance_from_optimality

def simulate_channel_switches(num_AP, sim_time=12*60*60, num_channels=4, scan_period=5, busy_factor=0.8, busy_factor_self=0.5, idle_factor=0.99, num_simulations=30, density=0.5, selection_periods_conditions=None):
    neighbor_matrix = generate_neighbor_matrix(num_AP, density)
    degrees = np.sum(neighbor_matrix, axis=1)
    
    results = []
    for condition in selection_periods_conditions:
        # Define selection period ranges based on degree
        selection_periods = np.zeros(num_AP, dtype=int)
        for i, degree in enumerate(degrees):
            if degree == 1:
                selection_periods[i] = condition[0]
            elif degree == 2:
                selection_periods[i] = condition[1]
            elif degree == 3:
                selection_periods[i] = condition[2]
            elif degree == 4:
                selection_periods[i] = condition[3]
            elif degree == 5:
                selection_periods[i] = condition[4]
            elif degree == 6:
                selection_periods[i] = condition[5]
            elif degree == 7:
                selection_periods[i] = condition[6]
            elif degree == 8:
                selection_periods[i] = condition[7]
            elif degree == 9:
                selection_periods[i] = condition[8]
            elif degree == 10:
                selection_periods[i] = condition[9]
            elif degree == 11:
                selection_periods[i] = condition[10]
            elif degree == 12:
                selection_periods[i] = condition[11]
            elif degree == 13:
                selection_periods[i] = condition[12]
            elif degree == 14:
                selection_periods[i] = condition[13]
            elif degree == 15:
                selection_periods[i] = condition[14]
            elif degree == 16:
                selection_periods[i] = condition[15]
            elif degree == 17:
                selection_periods[i] = condition[16]
            elif degree == 18:
                selection_periods[i] = condition[17]
            elif degree == 19:
                selection_periods[i] = condition[18]
            elif degree == 20:
                selection_periods[i] = condition[19]
            elif degree == 21:
                selection_periods[i] = condition[20]
            elif degree == 22:
                selection_periods[i] = condition[21]
            elif degree == 23:
                selection_periods[i] = condition[22]
            elif degree == 24:
                selection_periods[i] = condition[23]
            elif degree == 25:
                selection_periods[i] = condition[24]
            elif degree == 26:
                selection_periods[i] = condition[25]
            elif degree == 27:
                selection_periods[i] = condition[26]
            elif degree == 28:
                selection_periods[i] = condition[27]
            elif degree == 29:
                selection_periods[i] = condition[28]
            elif degree == 30:
                selection_periods[i] = condition[29]
            elif degree == 31:
                selection_periods[i] = condition[30]
            elif degree == 32:
                selection_periods[i] = condition[31]
            elif degree == 33:
                selection_periods[i] = condition[32]
            elif degree == 34:
                selection_periods[i] = condition[33]
            elif degree == 35:
                selection_periods[i] = condition[34]
            elif degree == 36:
                selection_periods[i] = condition[35]
            elif degree == 37:
                selection_periods[i] = condition[36]
            elif degree == 38:
                selection_periods[i] = condition[37]
            elif degree == 39:
                selection_periods[i] = condition[38]
            elif degree == 40:
                selection_periods[i] = condition[39]

        with mp.Pool(mp.cpu_count()) as pool:
            sim_results = pool.starmap(run_single_simulation, [(num_AP, sim_time, num_channels, scan_period, selection_periods, busy_factor, busy_factor_self, idle_factor, neighbor_matrix) for _ in range(num_simulations)])
        average_channel_switches = np.mean([result[0] for result in sim_results])
        std_channel_switches = np.std([result[0] for result in sim_results])
        total_distance_from_optimality = np.sum([result[1] for result in sim_results], axis=0)
        results.append((average_channel_switches, std_channel_switches, total_distance_from_optimality))
    
    return results

def generate_linear_periods(start, step, count):
    return [start + step * i for i in range(count)]

selection_periods_conditions = [
    generate_linear_periods(15 * 60, 15 * 60, 40),  # Original conditions
    generate_linear_periods(10 * 60, 10 * 60, 40),  # Shorter periods
    generate_linear_periods(20 * 60, 20 * 60, 40),  # Longer periods
    generate_linear_periods(5 * 60, 5 * 60, 40),     # Very short periods
    generate_linear_periods(15 * 60, 1 * 60, 40)     # Very short periods
]

num_APs = range(2, 41, 2)
sim_time = 36 * 60 * 60  # Define sim_time here
density = 0.5  # Define density here

all_results = []
for num_AP in num_APs:
    # Set the random seed before each simulation to ensure the same random draw
    np.random.seed(42)
    results = simulate_channel_switches(num_AP, sim_time=sim_time, density=density, selection_periods_conditions=selection_periods_conditions)
    all_results.append(results)

# Plot average channel switches
plt.figure()
for i, condition in enumerate(selection_periods_conditions):
    average_switches = [result[i][0] for result in all_results]
    std_switches = [result[i][1] for result in all_results]
    plt.errorbar(num_APs, average_switches, yerr=std_switches, marker='o', label=f'Condition {i+1}')

plt.xlabel('Number of APs')
plt.ylabel('Average Number of Channel Switches')
plt.title('Average Channel Switches vs Number of APs')
plt.legend()
plt.grid(True)
plt.show()

# Plot total distance from optimality
plt.figure()
for i, condition in enumerate(selection_periods_conditions):
    avg_total_distance = np.mean([result[i][2] for result in all_results], axis=0)
    plt.plot(np.arange(sim_time) / 60, avg_total_distance, label=f'Condition {i+1}')

plt.xlabel('Time (minutes)')
plt.ylabel('Total Distance from Optimality')
plt.title('Total Distance from Optimality over Time')
plt.legend()
plt.grid(True)
plt.show()