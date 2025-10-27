import os
import glob

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # File patterns to delete
    patterns = ["*.pdf", "*.png", "*.pgf"]

    for pattern in patterns:
        for file_path in glob.glob(os.path.join(script_dir, pattern)):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

if __name__ == "__main__":
    main()
