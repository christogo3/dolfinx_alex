# Python script to extract the maximum 'Jx' value from lines like 'Jx: <value> Jy: <value> Jz: <value>'

def find_max_jx_value(file_path):
    max_jx_value = float('-inf')  # Initialize with the smallest possible value

    with open(file_path, 'r') as file:
        for line in file:
            # Look for 'Jx:' in the line
            if 'Jx:' in line:
                # Split the line by spaces
                parts = line.split()
                # Loop through the parts to find 'Jx:'
                for part in parts:
                    if parts[0].startswith('Jx:'):
                        # Extract the value after 'Jx:'
                        # print(parts[1])
                        jx_value_str = parts[1]  # Get the value part and remove extra spaces
                        try:
                            jx_value = float(jx_value_str)  # Convert the extracted string to a float
                            max_jx_value = max(max_jx_value, jx_value)  # Update max_jx_value if the current one is larger
                        except ValueError:
                            print(f"Warning: Could not convert {jx_value_str} to float")

    return max_jx_value


# Example usage
file_path = 'bench.out.46523771'  # Replace with your file's path
max_value = find_max_jx_value(file_path)
print(f"The maximum value is: {max_value}")