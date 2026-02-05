import numpy as np
import zarr
a = zarr.open("madurai_30m.zarr")
def inspect_landsat_day(day_index):
    """
    Prints up to 10 non-NaN values from the 4D array for a specific day.
    """
    # 1. Slice the data for the specific day
    # Shape will be (2, 1610, 1724)
    try:
        day_slice = a["labels_30m/landsat/data"][day_index]
    except IndexError:
        print(f"Error: Day index {day_index} is out of range.")
        return

    # 2. Find indices where data is NOT NaN
    # Returns a tuple of (bands, rows, cols)
    valid_coords = np.where(~np.isnan(day_slice))
    
    num_found = len(valid_coords[0])
    
    if num_found == 0:
        print(f"Day {day_index}: All values are NaN.")
        return

    print(f"Day {day_index}: Found {num_found} valid pixels. Printing up to 10 samples:")
    print("-" * 60)
    
    # 3. Pull up to 10 values
    for i in range(min(10, num_found)):
        b = valid_coords[0][i]
        r = valid_coords[1][i]
        c = valid_coords[2][i]
        value = day_slice[b, r, c]
        
        # Printing as 4D coordinate: (Day, Band, Row, Col)
        print(f"Coord: ({day_index}, {b}, {r}, {c}) | Value: {value}")

# --- Usage ---
# To check day 0:
for i in range(4029):
    print(f"Day {i+1}")
    inspect_landsat_day(i)

# If day 0 was empty, try another day (e.g., mid-dataset):
# inspect_landsat_day(2000)
