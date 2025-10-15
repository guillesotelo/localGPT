import os

# Function to check if a line starts an exclusion block
def starts_exclusion_block(line):
    return line.startswith("```{eval-rst}") or line.startswith("```{toctree}")

# Function to check if a line ends an exclusion block
def ends_exclusion_block(line):
    return line.strip() == "```"

def concatenate_md_files(src_dir, dst_file):
    with open(dst_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.md') or file.endswith('.txt') or file.endswith('.rst'):
                    src_file = os.path.join(root, file)
                    with open(src_file, 'r', encoding='utf-8') as infile:
                        inside_block = False  # Track if we are inside an exclusion block
                        for line in infile:
                            # If we are inside an exclusion block, check for the closing ```
                            if inside_block:
                                if ends_exclusion_block(line):
                                    inside_block = False  # End of block
                                continue  # Skip all lines inside the block
                            
                            # If a new exclusion block starts, set inside_block to True
                            if starts_exclusion_block(line):
                                inside_block = True
                                continue  # Skip the start line of the block
                            
                            # Skipping images (in-line)
                            if(line.startswith('![')):
                                continue

                            # Write the line if we are not inside an exclusion block
                            outfile.write(line)
                        outfile.write("\n\n")  # Add a newline for separation between files
                    print(f"Copied content from {src_file}")

if __name__ == "__main__":
    source_directory = input("Enter the source directory path: ")
    destination_file = input("Enter the destination file path (including file extension): ")

    concatenate_md_files(source_directory, destination_file)
    print(f"All text files have been concatenated into {destination_file}.")