# 1. 对efs值进行分类，分为两类，使用字典数据结构。 {ID : (efs, efs_time)}
# 2. 对两个分类进行从小到大归并排序，merge sort
# 3. 合并二组为一个字典
# 4. 决定风险值：
#   4.1 若efs = 1, 直接决定
#   4.2 若efs = 0, 且前后若干点范围内皆存在一个efs = 1的值，则根据其位置的比重(欧式距离)确定风险值
#   4.3 若efs = 0, 且其位于最后一个efs = 1的点之后，则根据其与最后一个efs = 1的点和efs_time最大值的点(即risk_score = 0)之间的欧式距离确定其风险值、
# 5. 输出最终结果，数据结构为字典，键为ID，值为风险值。 {ID : risk_score}

import csv

from numexpr.necompiler import double


# parse csv file
def csv_parser(file_path):
    result = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            result[int(row['ID'])] = {'efs': double(row['efs']), 'efs_time': double(row['efs_time'])}
    return result

# calculate risk score
def risk_score(sorted_dict):
    risk_score_dict = {}
    for key in sorted_dict:
        if sorted_dict[key]['efs'] == 1:
            risk_score_dict[key] = 1 - key / len(sorted_dict)
    # for key in sorted_dict:
    #     if sorted_dict[key]['efs'] == 0:

    return risk_score_dict



def sort_efs_time(group):
    sorted_group = sorted(group.items(), key=lambda x: x[1]['efs_time'])
    #print(group_ids)

    result = {}
    for item in sorted_group:
        result[item[0]] = item[1]
    #print(result)

    return result


#test csv parser
#print(csv_parser('train.csv')[1])

#parse csv file
data = csv_parser('train.csv')
#divide data into two groups based on efs value
data_efs1 = {}
data_efs0 = {}
for key in data:
    if data[key]['efs'] == 1:
        data_efs1[key] = data[key]
    else:
        data_efs0[key] = data[key]
#print(data_efs0)
data_efs1_idx = list(sort_efs_time(data_efs1).keys())
data_efs0_idx = list(sort_efs_time(data_efs0).keys())
sorted_data = sort_efs_time(data)
sorted_data_idx = list(sorted_data.keys())
print(len(sorted_data))
print("amount of efs1: ", len(data_efs1))
#print(data_efs0_idx)
#print(len(data_efs1_idx))

risk_score_dict = {}
for key in sorted_data:
    if sorted_data[key]['efs'] == 1:
        #risk_score_dict[key] = 1 - key / len(sorted_data) #use index to calculate risk score
        risk_score_dict[key] = 1 - sorted_data[key]['efs_time'] / sorted_data[data_efs1_idx[-1]]['efs_time'] #use efs_time to calculate risk score
print(len(risk_score_dict))
for idx in range(len(sorted_data_idx)):
    if sorted_data[sorted_data_idx[idx]]['efs'] == 1 and sorted_data_idx[idx] != data_efs1_idx[-1]:
        # if sorted_data_idx[idx] == data_efs1_idx[-2]:
        #     print("!")
        step_to_next_efs1 = 1
        efs0_count = 0
        while sorted_data[sorted_data_idx[idx + step_to_next_efs1]]['efs'] != 1:
            step_to_next_efs1 += 1
            efs0_count += 1
        # if efs0_count == 0:
        #     continue
        next_efs1_idx = sorted_data_idx[idx + step_to_next_efs1]
        for i in range(1,efs0_count+1):
            current_id = sorted_data_idx[idx + i]
            current_efstime = sorted_data[current_id]['efs_time']
            previous_efstime = sorted_data[sorted_data_idx[idx]]['efs_time']
            next_efstime = sorted_data[next_efs1_idx]['efs_time']
            position_weight = (current_efstime - previous_efstime) / (next_efstime - previous_efstime)

            #risk_score_dict[sorted_data_idx[idx + i]] = (position_weight * risk_score_dict[sorted_data_idx[idx]] + (1-position_weight) * risk_score_dict[next_efs1_idx]) / 2
            risk_score_dict[sorted_data_idx[idx + i]] = position_weight * risk_score_dict[sorted_data_idx[idx]] + ((1 - position_weight) * risk_score_dict[next_efs1_idx] - position_weight * risk_score_dict[sorted_data_idx[idx]]) / 2
    #elif sorted_data[sorted_data_idx[idx]]['efs'] == 1 and sorted_data_idx[idx] == data_efs1_idx[-1]:
    elif sorted_data[sorted_data_idx[idx]]['efs'] == 1:
        efs0_count = len(sorted_data_idx) - idx - 1
        # while sorted_data[sorted_data_idx[idx + efs0_count]]['efs'] != 1:
        #     efs0_count += 1

        for i in range(1,efs0_count+1):
            current_idx = sorted_data_idx[idx + i]
            current_efstime = sorted_data[current_idx]['efs_time']
            previous_efstime = sorted_data[sorted_data_idx[idx]]['efs_time']
            max_efstime = sorted_data[sorted_data_idx[-1]]['efs_time']
            position_weight = (current_efstime - previous_efstime) / (max_efstime - previous_efstime)
            risk_score_dict[sorted_data_idx[idx + i]] = (position_weight * risk_score_dict[sorted_data_idx[idx]]) / 2

print(risk_score_dict)
risk_score_dict = dict(sorted(risk_score_dict.items()))
#print(risk_score_dict)
#print(data)
print(len(risk_score_dict))

prev = -1
for key in risk_score_dict:
    if key - prev != 1:
        print(key)
    prev = key
#print(sorted_data)
