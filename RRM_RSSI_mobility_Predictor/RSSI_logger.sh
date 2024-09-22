#!/bin/bash

# Define the output CSV file
output_file="rssi_log.csv"

# Write the CSV header
echo "Timestamp,Interface,Station MAC,Signal (dBm),Last ACK Signal (dBm),Avg ACK Signal (dBm)" > "$output_file"

# Define the duration and interval
duration=$((8 * 60 * 60))  # 8 hours in seconds
interval=1  # Log every 1 second 

# Start the logging process
start_time=$(date +%s)

while [ $(($(date +%s) - start_time)) -lt "$duration" ]; do
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
    
    # Wait for the next interval
    sleep "$interval"
done

echo "Logging completed. Data saved in $output_file."
