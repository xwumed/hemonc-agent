import os
import shutil

def flatten_pdfs(src_dir, dst_dir):
    # Create destination folder if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        # Skip the root folder itself to only process subfolders
        if os.path.abspath(root) == os.path.abspath(src_dir):
            continue

        # Use the subfolder name as prefix
        subfolder = os.path.basename(root)
        for filename in files:
            if filename.lower().endswith('.pdf'):
                src_path = os.path.join(root, filename)
                prefix = f"{subfolder}_"
                new_name = prefix + filename
                dst_path = os.path.join(dst_dir, new_name)

                # Handle potential name collisions
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while True:
                        new_name = f"{subfolder}_{base}_{counter}{ext}"
                        dst_path = os.path.join(dst_dir, new_name)
                        if not os.path.exists(dst_path):
                            break
                        counter += 1

                # Copy the file to the destination
                shutil.copy2(src_path, dst_path)

    print(f"All PDFs have been copied and flattened into '{dst_dir}'")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Flatten PDFs from subfolders into a single folder with subfolder name prefixes.'
    )
    parser.add_argument(
        'source',
        help='Path to the main folder containing PDF subfolders'
    )
    parser.add_argument(
        'destination',
        help='Path to the new folder where flattened PDFs will be stored'
    )
    args = parser.parse_args()

    flatten_pdfs(args.source, args.destination)
