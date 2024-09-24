import numpy as np
import matplotlib.pyplot as plt

# Parameters
sim_time = 12 * 60 * 60  # in seconds
num_channels = 4  # non-DFS channels
num_AP = 10
scan_period = 5  # time in seconds
selection_period = 15 * 60  # in seconds

busy_factor = 0.8
busy_factor_self = 0.5
idle_factor = 0.99

# Traffic parameters
lambda_on_AP = 0.1 * np.ones(num_AP)
lambda_off_AP = 0.0001 * np.ones(num_AP)
max_load_AP = np.random.rand(num_AP)

# Start switch
start_sel = np.ceil(selection_period * np.random.rand(num_AP)).astype(int)

# Initialization
cahnnel_busy_AP = np.zeros((num_AP, num_channels))
g_channel_busy = np.zeros((num_AP, num_channels))
g_cur_channel_busy = np.zeros(num_AP)
channels = np.ones(num_AP, dtype=int)
scan_ch = np.ones(num_AP, dtype=int)

channels_time = np.zeros((num_AP, sim_time))

# Generate Traffic - On off model
traffic = np.zeros((num_AP, sim_time))
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

# Collect points where (time + start_sel[AP]) % selection_period == 0
selection_points = {AP: [] for AP in range(num_AP)}

# Metric for distance from optimality
distance_from_optimality = np.zeros((num_AP, sim_time))

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
            selection_points[AP].append(time)
            ch = channels[AP]
            busy = np.sum(np.mean((channels[:, np.newaxis] == ch) * traffic[:, max(0, time - selection_period):time], axis=1))
            tx = np.mean(traffic[AP, max(0, time - selection_period):time])
            busy -= tx
            g_cur_channel_busy[AP] = g_cur_channel_busy[AP] * busy_factor_self + (1 - busy_factor_self) * busy
            g_channel_busy[AP, ch - 1] = g_cur_channel_busy[AP]
            min_avg = np.min(g_channel_busy[AP, :])
            best_ch = np.argmin(g_channel_busy[AP, :]) + 1
            if g_cur_channel_busy[AP] > 1.25 * min_avg and g_cur_channel_busy[AP] > 0.15:
                # Move channel
                channels[AP] = best_ch
                g_channel_busy[AP, :] = np.zeros(num_channels)
                g_cur_channel_busy[AP] = 0
        # Calculate distance from optimality
        current_ch = channels[AP] - 1
        if g_channel_busy[AP, current_ch] > np.min(g_channel_busy[AP, :]):
            distance_from_optimality[AP, time] = 1
    channels_time[:, time] = channels

# Plotting
plt.figure(1)
plt.plot(np.arange(sim_time) / 60, traffic[0, :])
plt.xlabel('Time (minutes)')
plt.ylabel('Traffic for AP 1')
plt.title('Traffic over Time for AP 1')
plt.show()

plt.figure(2)
plt.plot(np.arange(sim_time) / 60, channels_time.T)
plt.xlabel('Time (minutes)')
plt.ylabel('Channel')
plt.title('Channel Selection over Time for all APs')

# Add points where (time + start_sel[AP]) % selection_period == 0
for AP in range(num_AP):
    times = np.array(selection_points[AP]) / 60  # Convert to minutes
    plt.scatter(times, channels_time[AP, selection_points[AP]], s=10, label=f'AP {AP+1} Selection Points')

plt.legend()
plt.show()

# Plot distance from optimality
plt.figure(3)
# Sum the distance from optimality across all APs
total_distance_from_optimality = np.sum(distance_from_optimality, axis=0)

# Plot the total distance from optimality
plt.figure()
plt.plot(np.arange(sim_time) / 60, total_distance_from_optimality, label='Total Distance from Optimality')
plt.xlabel('Time (minutes)')
plt.ylabel('Total Distance from Optimality')
plt.title('Total Distance from Optimality over Time')
plt.legend()
plt.show()