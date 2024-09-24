#!/usr/bin/env ucode

global.fs = require('fs');
global.math = require('math');

// Manually trim leading and trailing whitespace
function manual_trim(str) {
    let trimmedStr = "";
    let leadingWhitespace = true;
    let i = 0;

    while (true) {
        if (!(i in str)) break;
        let char = str[i];

        if (leadingWhitespace && (char == ' ' || char == '\t' || char == '\r')) {
            i = math.add(i, 1);
            continue;
        }

        leadingWhitespace = false;
        trimmedStr += char;
        i = math.add(i, 1);
    }

    let endIdx = 0;
    let finalStr = "";
    let j = 0;
    while (j in trimmedStr) {
        let char = trimmedStr[j];
        if (char != ' ' && char != '\t' && char != '\r') {
            endIdx = j;
        }
        j = math.add(j, 1);
    }

    let k = 0;
    while (k <= endIdx) {
        finalStr += trimmedStr[k];
        k = math.add(k, 1);
    }

    return finalStr;
}

// Write to a file
function write_to_file(filepath, content) {
    let file = fs.open(filepath, 'r+'); 
    if (!file) {
        print("ERROR: Unable to open file " + filepath + " for writing.");
        return;
    }

    let existingContent = file.read("all") || "";
    file.close();

    file = fs.open(filepath, 'w');
    if (!file) {
        print("ERROR: Unable to open file " + filepath + " for writing.");
        return;
    }

    let newContent = existingContent + content; 
    file.write(newContent);
    file.close();
    print("DEBUG: Wrote to file " + filepath);
}

// Manually split a string with a comma delimiter
function manual_split_func(str, delimiter) {
    print("DEBUG: Attempting to split using delimiter '" + delimiter + "'");
    let result = [];
    let currentField = "";
    let i = 0;
    
    while (true) {
        if (!(i in str)) break;
        let char = str[i];

        // Check if char matches the delimiter character by character
        if (char == delimiter) {
            print("DEBUG: Detected delimiter at position " + i + ", currentField: " + currentField);
            result.push(currentField);
            currentField = "";
        } else {
            currentField += char;
        }
        i = math.add(i, 1);
    }

    // Add the last field
    if (currentField != "") {
        result.push(currentField);
        print("DEBUG: Adding final field: " + currentField);
    }

    print("DEBUG: Total fields split: " + result);
    return result;
}

// Read the last entry from a CSV file
function read_last_entry(filepath) {
    print("DEBUG: Attempting to open file: " + filepath);
    let file = fs.open(filepath, 'r');
    if (!file) {
        print("ERROR: Failed to open file " + filepath);
        return null;
    }

    let allContent = file.read("all");
    file.close();

    if (!allContent) {
        print("ERROR: No content found in file " + filepath);
        return null;
    }

    print("DEBUG: Content read from file:\n" + allContent);

    let lines = manual_split_func(allContent, "\n");
    let totalLines = 0;
    
    // Count total lines manually
    while (totalLines in lines) {
        totalLines = math.add(totalLines, 1);
    }

    print("DEBUG: Total lines found: " + totalLines);

    let lastNonEmptyLine = "";
    let i = 0;

    while (i in lines) {
        let currentLine = lines[i];
        print("DEBUG: Current line being processed: " + currentLine);

        let trimmedLine = manual_trim(currentLine);
        print("DEBUG: Trimmed line: " + trimmedLine);

        if (trimmedLine != "") {
            print("DEBUG: Found a valid non-empty trimmed line: " + trimmedLine);
            lastNonEmptyLine = trimmedLine;
        }
        i = math.add(i, 1);
    }

    if (lastNonEmptyLine == "") {
        print("ERROR: No valid data lines found in RSSI log.");
        return null;
    }

    print("DEBUG: Last valid non-empty line found: " + lastNonEmptyLine);
    return lastNonEmptyLine;
}

// Main training function
function train_model() {
    let RSSI_LOG = "/etc/rssi_log.csv";
    let EVAL_FILE = "/etc/rssi_evaluation.csv";

    write_to_file(EVAL_FILE, "Timestamp,Station MAC,Actual RSSI,Predicted 1 sec\n");

    function process_epoch() {
        let latest_entry = read_last_entry(RSSI_LOG);
        if (!latest_entry) {
            print("ERROR: No valid data found in RSSI log.");
            return;
        }

        let fields = manual_split_func(latest_entry, ',');
        let fieldCount = 0;

        // Count fields manually
        while (fieldCount in fields) {
            fieldCount = math.add(fieldCount, 1);
        }

        print("DEBUG: Fields extracted: " + fields);

        if (fieldCount < 6) {
            print("ERROR: Insufficient fields in data.");
            return;
        }

        let rssiString = manual_trim(fields[5]);
        rssiString = rssiString.replace("dBm", ""); 

        let rssiValue = tonumber(rssiString);

        if (rssiValue == null || isNaN(rssiValue)) {
            print("ERROR: Invalid RSSI value found: " + rssiString);
            return;
        }

        let timestamp = fields[0];
        let station = fields[2];

        print("INFO: Starting training for station " + station + " at " + timestamp);

        let epoch = 1;
        while (epoch <= 10) {
            print("Training progress - Epoch " + epoch + ": Station = " + station + ", RSSI = " + rssiValue);
            
            let predictedRSSI = math.add(rssiValue, epoch);

            let resultLine = timestamp + "," + station + "," + rssiValue + "," + predictedRSSI + "\n";
            write_to_file(EVAL_FILE, resultLine);

            print("Result written: " + resultLine);
            epoch = math.add(epoch, 1);
        }
    }

    process_epoch();
}

// Start training
train_model();
