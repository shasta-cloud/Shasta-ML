#!/bin/bash

# Define the output CSV file
output_file="rssi_log.csv"
temp_file="rssi_log_temp.csv"

# Write the CSV header
echo "Timestamp,Interface,Station MAC,Signal (dBm),Last ACK Signal (dBm),Avg ACK Signal (dBm)" > "$output_file"

# Define the interval
interval=1  # Log every 1 second 

# Function to check and maintain FIFO log size
maintain_fifo_log() {
    local max_lines=10000
    local current_lines=$(wc -l < "$output_file")
    if [ "$current_lines" -gt "$max_lines" ]; then
        tail -n "$max_lines" "$output_file" > "$temp_file"
        mv "$temp_file" "$output_file"
    fi
}

# Function to remove constant RSSI values
remove_constant_rssi() {
    awk -F, '
    BEGIN {
        OFS = FS
    }
    NR == 1 {
        print $0
        next
    }
    {
        mac = $3
        signal = $4
        if (mac in last_signal) {
            if (signal < last_signal[mac] - 5 || signal > last_signal[mac] + 5) {
                print $0
                last_signal[mac] = signal
            }
        } else {
            print $0
            last_signal[mac] = signal
        }
    }' "$output_file" > "$temp_file"
    mv "$temp_file" "$output_file"
}

# Start the logging process
while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Iterate through all wireless interfaces
    for iface in $(iw dev | awk '$1=="Interface"{print $2}'); do
        # Get station dump information for each interface
        iw dev "$iface" station dump | awk -v iface="$iface" -v ts="$timestamp" '
            /Station/ {mac=$2}
            /signal:/ {signal=$2}
            /last ack signal:/ {last_ack=$4}
            /avg ack signal:/ {avg_ack=$4; 
            print ts "," iface "," mac "," signal "," last_ack "," avg_ack}
        ' >> "$output_file"
    done
    
    # Maintain FIFO log size
    maintain_fifo_log
    
    # Remove constant RSSI values
    remove_constant_rssi
    
    # Wait for the next interval
    sleep "$interval"
done