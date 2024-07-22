#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 start_number end_number"
    exit 1
}

# Check if exactly 2 arguments are provided
if [ "$#" -ne 2 ]; then
    usage
fi

# Check if the provided arguments are integers
if ! [[ "$1" =~ ^[0-9]+$ ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo "Error: Both arguments must be positive integers."
    usage
fi

# Assign input arguments to variables
start_number=$1
end_number=$2

# Check if start_number is less than or equal to end_number
if [ "$start_number" -gt "$end_number" ]; then
    echo "Error: Start number must be less than or equal to end number."
    usage
fi

# Loop through the range and cancel each job
for (( job_id=$start_number; job_id<=$end_number; job_id++ ))
do
    echo "Cancelling job ID $job_id"
    scancel "$job_id"
done

echo "All jobs in the range $start_number to $end_number have been cancelled."
