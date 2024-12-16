import os

def add_prefix_to_files(prefix, directory="/home/yifan/Documents/data/forest/test/VH"):
    """
    Add a specified prefix to all files in the current directory.
    
    Args:
    - prefix (str): The string to add before the file names.
    - directory (str): The directory where file names will be modified. Defaults to the current directory.
    """
    # Get all files in the directory
    for filename in os.listdir(directory):
        # Check if it is a file (ignore directories)
        if os.path.isfile(os.path.join(directory, filename)):
            # New file name: add the prefix to the existing file name
            new_filename = prefix + filename
            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed file {filename} to {new_filename}")

# Example: Add "prefix_" to all file names in the current directory
if __name__ == "__main__":
    # Input prefix
    prefix = input("Please enter the prefix to add: ")
    # Execute file name modification
    add_prefix_to_files(prefix)