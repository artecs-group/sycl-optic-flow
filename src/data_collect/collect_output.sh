#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <number_of_executions> <output_file> <command_to_execute>"
    exit 1
fi

# Extract the arguments
num_executions="$1"
output_file="$2"
command_to_execute="$3"

# Initialize the output file
> "$output_file"

# Loop to execute the program and collect output
for ((i = 1; i <= num_executions; i++)); do
    echo "Executing command: $command_to_execute (Run $i)"
    # Run the program and append its output to the output file
    $command_to_execute >> "$output_file" 2>&1

    # Add a separator line after each execution
    echo "-------------------" >> "$output_file"

    if [ $? -ne 0 ]; then
        echo "Error: Command failed on Run $i"
        exit 1
    fi
done

echo "Execution completed $num_executions times. Output saved to $output_file"
