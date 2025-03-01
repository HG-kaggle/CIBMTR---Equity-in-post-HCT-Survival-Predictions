import csv
import tensorflow as tf

# parse csv file
def csv_parser(file_path):
    result = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            result[int(row['ID'])] = {
                'efs': tf.cast(int(row['efs']), dtype=tf.int32),
                'efs_time': tf.convert_to_tensor(float(row['efs_time']), dtype=tf.float32)
            }
    return result

# Sorting function
def sort_efs_time(group):
    sorted_group = sorted(group.items(), key=lambda x: x[1]['efs_time'].numpy())  # Convert tensor to numpy for sorting
    return {item[0]: item[1] for item in sorted_group}

# Parse CSV file
data = csv_parser('train.csv')

# Divide data into two groups based on efs value
data_efs1 = {key: data[key] for key in data if data[key]['efs'] == 1}
data_efs0 = {key: data[key] for key in data if data[key]['efs'] == 0}

# Sort by efs_time
sorted_data = sort_efs_time(data)
sorted_data_idx = list(sorted_data.keys())
data_efs1_idx = list(sort_efs_time(data_efs1).keys())

# Risk score calculation
risk_score_dict = {}
for key in sorted_data:
    if sorted_data[key]['efs'] == 1:
        risk_score_dict[key] = 1 - sorted_data[key]['efs_time'] / sorted_data[data_efs1_idx[-1]]['efs_time']

# Fill risk scores for efs=0 using interpolation
for idx in range(len(sorted_data_idx)):
    if sorted_data[sorted_data_idx[idx]]['efs'] == 1:
        step_to_next_efs1 = 1
        efs0_count = 0
        while (idx + step_to_next_efs1) < len(sorted_data_idx) and sorted_data[sorted_data_idx[idx + step_to_next_efs1]]['efs'] != 1:
            step_to_next_efs1 += 1
            efs0_count += 1

        next_efs1_idx = sorted_data_idx[idx + step_to_next_efs1] if idx + step_to_next_efs1 < len(sorted_data_idx) else sorted_data_idx[-1]

        for i in range(1, efs0_count + 1):
            current_idx = sorted_data_idx[idx + i]
            current_efstime = sorted_data[current_idx]['efs_time']
            previous_efstime = sorted_data[sorted_data_idx[idx]]['efs_time']
            next_efstime = sorted_data[next_efs1_idx]['efs_time']
            position_weight = (current_efstime - previous_efstime) / (next_efstime - previous_efstime)

            risk_score_dict[current_idx] = position_weight * risk_score_dict[sorted_data_idx[idx]] + ((1 - position_weight) * risk_score_dict[next_efs1_idx] - position_weight * risk_score_dict[sorted_data_idx[idx]]) / 2

# Convert risk scores to TensorFlow tensors
risk_score_dict = {key: tf.convert_to_tensor(value, dtype=tf.float32) for key, value in risk_score_dict.items()}

print(risk_score_dict)
