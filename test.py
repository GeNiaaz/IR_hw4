def process_AND(list_a, list_b):
    ptr_a = 0
    ptr_b = 0

    max_index_a = len(list_a)
    max_index_b = len(list_b)

    resultant_list = []

    while ptr_a < max_index_a and ptr_b < max_index_b:
        print("a: ", ptr_a)
        print("b: ", ptr_b)
        curr_a = list_a[ptr_a]
        curr_b = list_b[ptr_b]

        if curr_a == curr_b:
            ptr_a += 1
            ptr_b += 1

            resultant_list.append(curr_a)
        else:
            if curr_a < curr_b:
                ptr_a += 1
            else:
                ptr_b += 1

    print(resultant_list)
    return resultant_list


ls1 = [1,2,3,4,5,6,7,8,10]
ls2 = [2,4,6,7,9,10]

if __name__ == '__main__':
    process_AND(ls1, ls2)
