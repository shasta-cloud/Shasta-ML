#!/bin/sh

# Function to manually trim whitespace from strings
manual_trim() {
    str="$1"
    trimmedStr=""
    leadingWhitespace=true
    i=0
    length=${#str}
    echo "DEBUG: String length: $length"

    while [ $i -lt $length ]; do
        char=$(printf "%s" "$str" | cut -c $((i + 1)))
        
        if [ "$char" != " " ] && [ "$char" != "\t" ]; then
            leadingWhitespace=false
        fi
        
        if [ "$leadingWhitespace" = false ]; then
            trimmedStr="$trimmedStr$char"
        fi
        
        i=$((i + 1))
    done

    # Remove trailing whitespace
    echo "$trimmedStr" | sed 's/[ \t]*$//'
}

# Function to manually split a string by a delimiter
manual_split_func() {
    input="$1"
    delimiter="$2"
    fields=""
    currentField=""
    i=1
    input_length=${#input}
    
    while [ $i -le $input_length ]; do
        char=$(printf "%s" "$input" | cut -c $i)
        
        if [ "$char" = "$delimiter" ]; then
            fields="$fields$currentField "
            currentField=""
        else
            currentField="$currentField$char"
        fi
        
        i=$((i + 1))
    done
    
    # Add the last field
    fields="$fields$currentField"
    echo "$fields"
}

# Function to count elements in a space-separated list
manual_count() {
    list="$1"
    count=0
    
    for item in $list; do
        count=$((count + 1))
    done
    
    echo $count
}

# Function to read the last non-empty line from a CSV file
read_last_entry() {
    echo "DEBUG: Reading last non-empty line from file..."
    filepath="$1"
    if [ ! -f "$filepath" ]; then
        echo "DEBUG: File does not exist: $filepath"
        echo ""
        return
    fi
    echo "DEBUG: Attempting to open file: $filepath"
    file=$(cat "$filepath")
    
    if [ -z "$file" ]; then
        echo "DEBUG: File is empty"
        echo ""
        return
    fi
    
    lines=$(echo "$file" | tr '\n' '|')
    split_lines=$(manual_split_func "$lines" "|")
    
    lastNonEmptyLine=""
    for line in $split_lines; do
        echo "DEBUG: Processing line: $line"
        trimmed_line=$(manual_trim "$line")
        if [ ! -z "$trimmed_line" ]; then
            lastNonEmptyLine="$trimmed_line"
        fi
    done
    
    echo "DEBUG: Last non-empty line: $lastNonEmptyLine"
    echo "$lastNonEmptyLine"
}

# Function to write to a file
write_to_file() {
    filepath="$1"
    content="$2"
    echo "$content" > "$filepath"
    echo "DEBUG: Wrote to file $filepath"
}

# Main function
train_model() {
    RSSI_LOG="/etc/rssi_log.csv"
    EVAL_FILE="/etc/rssi_evaluation.csv"
    
    write_to_file "$EVAL_FILE" "Timestamp,Station MAC,Actual RSSI,Predicted 1 sec\n"
    
    echo "Starting ARIMA training..."

    process_epoch() {
        echo "Reading latest RSSI data..."
        latest_entry=$(read_last_entry "$RSSI_LOG")
        echo "DEBUG: Latest entry: $latest_entry"
        
        if [ -z "$latest_entry" ]; then
            echo "ERROR: No valid data found in RSSI log."
            return
        fi
        
        echo "DEBUG: Latest entry: $latest_entry"
        fields=$(manual_split_func "$latest_entry" ",")
        echo "DEBUG: Fields: $fields"
        count=$(manual_count "$fields")
        echo "DEBUG: Field count: $count"
        
        if [ "$count" -lt 6 ]; then
            echo "ERROR: Unexpected number of fields in the line: $latest_entry"
            return
        fi
        
        timestamp=$(echo "$fields" | cut -d ' ' -f 1)
        station=$(echo "$fields" | cut -d ' ' -f 2)
        rssiString=$(echo "$fields" | cut -d ' ' -f 6)
        rssiValue=$(echo "$rssiString" | sed 's/[^0-9-]*//g')

        echo "DEBUG: Reading line - Timestamp: $timestamp, Station MAC: $station, RSSI: $rssiValue"

        # Placeholder for ARIMA prediction logic
        predicted_rssi=$rssiValue

        echo "$timestamp,$station,$rssiValue,$predicted_rssi" >> "$EVAL_FILE"
        echo "DEBUG: Epoch completed. Predicted RSSI for $station: $predicted_rssi"
    }

    epochs=5
    for epoch in $(seq 1 $epochs); do
        echo "Starting epoch $epoch..."
        process_epoch
        echo "Epoch $epoch completed."
    done
}

# Start the ARIMA training process
train_model