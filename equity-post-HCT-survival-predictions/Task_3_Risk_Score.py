import csv
import tensorflow as tf

# Parse CSV file
def csv_parser(file_path):
    result = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            result[int(row['ID'])] = {
                'efs': int(float(row['efs'])),  # Ensure it's an integer
                'efs_time': float(row['efs_time'])  # Keep as float for sorting
            }
    return result

# Sorting function
def sort_efs_time(group):
    return dict(sorted(group.items(), key=lambda x: x[1]['efs_time']))  # Sort by efs_time

# Parse CSV file
data = csv_parser('train.csv')

# Divide data into two groups based on efs value
data_efs1 = {key: data[key] for key in data if data[key]['efs'] == 1}
data_efs0 = {key: data[key] for key in data if data[key]['efs'] == 0}

# Sort by efs_time
sorted_data = sort_efs_time(data)
sorted_data_idx = list(sorted_data.keys())
sorted_efs1 = sort_efs_time(data_efs1)
data_efs1_idx = list(sorted_efs1.keys())

# Risk score calculation
risk_score_dict = {}
if data_efs1_idx:  # Ensure there's at least one efs=1 entry
    max_efs_time = sorted_data[data_efs1_idx[-1]]['efs_time']  # Get max time for normalization

    # Ensure all efs=1 entries have risk scores
    for key in sorted_data:
        if sorted_data[key]['efs'] == 1:
            risk_score_dict[key] = 1 - sorted_data[key]['efs_time'] / max_efs_time

# Fill risk scores for efs=0 using stable interpolation
EPSILON = 1e-8  # Small constant to avoid division by zero

for idx, key in enumerate(sorted_data_idx):
    if sorted_data[key]['efs'] == 1:
        step_to_next_efs1 = 1
        efs0_count = 0

        # Find the next efs=1 in the sorted list
        while (idx + step_to_next_efs1) < len(sorted_data_idx) and sorted_data[sorted_data_idx[idx + step_to_next_efs1]]['efs'] != 1:
            step_to_next_efs1 += 1
            efs0_count += 1

        next_efs1_idx = sorted_data_idx[idx + step_to_next_efs1] if idx + step_to_next_efs1 < len(sorted_data_idx) else None

        prev_efstime = sorted_data[key]['efs_time']
        next_efstime = sorted_data[next_efs1_idx]['efs_time'] if next_efs1_idx else prev_efstime

        # Ensure no division by zero
        denom = max(next_efstime - prev_efstime, EPSILON)

        for i in range(1, efs0_count + 1):
            current_idx = sorted_data_idx[idx + i]
            current_efstime = sorted_data[current_idx]['efs_time']

            # Compute interpolation weight
            position_weight = (current_efstime - prev_efstime) / denom

            prev_risk = risk_score_dict.get(key, 0.0)  # Default to 0 if missing
            next_risk = risk_score_dict.get(next_efs1_idx, prev_risk)  # Default to prev_risk if missing

            # Compute interpolated risk score
            risk_score_dict[current_idx] = prev_risk + position_weight * (next_risk - prev_risk)

# Convert risk scores to TensorFlow tensors
risk_score_dict = {key: tf.convert_to_tensor(value, dtype=tf.float32) for key, value in risk_score_dict.items()}

print(risk_score_dict)
