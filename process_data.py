import os
import json

# File paths
input_file = 'dataset/clean_benchmarks/IndustryOR.jsonl'
output_dir = 'processed_dataset/IndustryOR'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for the total number of entries
total_entries = 0

# Read the original data
with open(input_file, 'r') as f:
    for index, line in enumerate(f, start=1):
        data = json.loads(line)

        # Extract information
        question = data['en_question']
        answer = data['en_answer']

        # Create a directory for the entry
        entry_dir = os.path.join(output_dir, str(index))
        os.makedirs(entry_dir, exist_ok=True)

        # Write the description.txt file
        description_file = os.path.join(entry_dir, 'description.txt')
        with open(description_file, 'w') as desc_file:
            desc_file.write(question)

        # Write the input_targets.json file
        input_targets_file = os.path.join(entry_dir, 'origin_format.json')
        with open(input_targets_file, 'w') as input_file:
            json.dump(data, input_file, ensure_ascii=False, indent=4)

        # Write the Answer.txt file
        answer_file = os.path.join(entry_dir, 'answer.txt')
        with open(answer_file, 'w') as ans_file:
            ans_file.write(answer)

        # Print success message for each entry
        print(f"Successfully processed entry {index}.")

        # Increment the total entries count
        total_entries += 1

# Print the total number of entries processed
print(f"Total entries processed: {total_entries}")
