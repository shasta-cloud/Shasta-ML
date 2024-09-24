import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def run_single_simulation(num_AP, sim_time, num_channels, scan_period, selection_period, busy_factor, busy_factor_self, idle_factor):
    lambda_on_AP = 0.1 * np.ones(num_AP)
    lambda_off_AP = 0.0001 * np.ones(num_AP)
    max_load_AP = np.random.rand(num_AP)
    start_sel = np.ceil(selection_period * np.random.rand(num_AP)).astype(int)
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
                    cahnnel_busy_AP[AP, ch - 1] = min(np.sum((channels == ch) * traffic[:, time]), 1)
                    if cahnnel_busy_AP[AP, ch - 1] > 0:
                        g_channel_busy[AP, ch - 1] = g_channel_busy[AP, ch - 1] * busy_factor + (1 - busy_factor) * cahnnel_busy_AP[AP, ch - 1]
                    else:
                        g_channel_busy[AP, ch - 1] = g_channel_busy[AP, ch - 1] * idle_factor
                scan_ch[AP] = (ch < num_channels) * (ch + 1) + (ch == num_channels) * 1
            if (time + start_sel[AP]) % selection_period == 0:
                ch = channels[AP]
                busy = np.sum(np.mean((channels[:, np.newaxis] == ch) * traffic[:, max(0, time - selection_period):time], axis=1))
                tx = np.mean(traffic[AP, max(0, time - selection_period):time])
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

def simulate_channel_switches(num_AP, selection_period, sim_time=12*60*60, num_channels=4, scan_period=5, busy_factor=0.8, busy_factor_self=0.5, idle_factor=0.99, num_simulations=30):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(run_single_simulation, [(num_AP, sim_time, num_channels, scan_period, selection_period, busy_factor, busy_factor_self, idle_factor) for _ in range(num_simulations)])
    average_channel_switches = np.mean([result[0] for result in results])
    std_channel_switches = np.std([result[0] for result in results])
    total_distance_from_optimality = np.sum([result[1] for result in results], axis=0)
    return average_channel_switches, std_channel_switches, total_distance_from_optimality

num_APs = range(2, 41, 2)
selection_periods = [15*60, 30*60, 45*60, 60*60, 120*60]   # Different selection periods in seconds
average_switches = {sp: [] for sp in selection_periods}
std_switches = {sp: [] for sp in selection_periods}
total_distances = {sp: [] for sp in selection_periods}

sim_time = 12 * 60 * 60  # Define sim_time here

for sp in selection_periods:
    for num_AP in num_APs:
        avg_switches, std_switch, total_distance = simulate_channel_switches(num_AP, sp, sim_time=sim_time)
        average_switches[sp].append(avg_switches)
        std_switches[sp].append(std_switch)
        total_distances[sp].append(total_distance)

for sp in selection_periods:
    plt.errorbar(num_APs, average_switches[sp], yerr=std_switches[sp], marker='o', label=f'Selection Period {sp//60} min')

plt.xlabel('Number of APs')
plt.ylabel('Average Number of Channel Switches')
plt.title('Average Channel Switches vs Number of APs')
plt.legend()
plt.grid(True)
plt.show()

# Plot total distance from optimality
plt.figure()
for sp in selection_periods:
    avg_total_distance = np.mean(total_distances[sp], axis=0)
    plt.plot(np.arange(sim_time) / 60, avg_total_distance, label=f'Selection Period {sp//60} min')

plt.xlabel('Time (minutes)')
plt.ylabel('Total Distance from Optimality')
plt.title('Total Distance from Optimality over Time')
plt.legend()
plt.show()