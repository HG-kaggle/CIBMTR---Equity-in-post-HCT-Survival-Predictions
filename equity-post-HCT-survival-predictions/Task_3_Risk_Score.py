# 1. 对efs值进行分类，分为两类，使用字典数据结构。 {ID : (efs, efs_time)}
# 2. 对两个分类进行从小到大归并排序，merge sort
# 3. 合并二组为一个字典
# 4. 决定风险值：
#   4.1 若efs = 1, 直接决定
#   4.2 若efs = 0, 且前后若干点范围内皆存在一个efs = 1的值，则根据其位置的比重(欧式距离)确定风险值
#   4.3 若efs = 0, 且其位于最后一个efs = 1的点之后，则根据其与最后一个efs = 1的点和efs_time最大值的点(即risk_score = 0)之间的欧式距离确定其风险值、
# 5. 输出最终结果，数据结构为字典，键为ID，值为风险值。 {ID : risk_score}

# calculate risk score
def risk_score(sorted_dict):

    return

# merge two sorted dictionaries
def merge_group(group_1, group_2):

    return

# merge sort
def merge_sort(group):
    #if the length of the dictionary is less than or equal to 1, return this dictionary (base case)
    if len(group) <= 1:
        return group
    mid = int(len(group) / 2) + group.keys()[0] #calculate the key of middle element
    #devide the dictionary into right and left part and call this method recursively
    left = merge_sort({k : v for (k, v) in group if k < mid})
    right = merge_sort({k : v for (k, v) in group if k >= mid})
    return merge_sort_helper(left, right)

def merge_sort_helper(left, right):
    r_index = 0 #right index pointer
    l_index = 0 #left index pointer
    result = {} #result dict
    #compare and add to result dict
    while l_index < len(left) and r_index < len(right):
        left_key = left.keys()[l_index] #key that the left pointer points to
        right_key = right.keys()[r_index] #key that the right pointer points to
        #compare
        if left[left_key] <= right[right_key]: #if left one is smaller, add it to result
            result[left_key] = left[left_key]
            l_index += 1 #increase left pointer value by 1
        else: #if right one is smaller, add it to result
            result[right_key] = right[right_key]
            r_index += 1 #increase right pointer value by 1

    #add remaining value in left or right into result
    while l_index < len(left):
        left_key = left.keys()[l_index]
        result[left_key] = left[left_key]
        l_index += 1
    while r_index < len(right):
        right_key = right.keys()[r_index]
        result[right_key] = right[right_key]
        r_index += 1

    return result #return result
