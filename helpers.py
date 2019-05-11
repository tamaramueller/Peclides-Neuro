def string_to_int_list(my_string):
    str_list = [x.strip() for x in my_string.split(',')]
    return map(int, str_list)


def compare_two_lists(list1, list2):
    res = []
    if len(list1) == len(list2):
        for i in range(0, len(list1)):
            if list1[i] == list2[i]:
                res.append(1)
            else:
                res.append(0)
    return res


def get_correctly_classified_2(vector, ytest):
    res = []
    for i in range(0, len(vector)):
        if vector[i] == ytest[i]:
            res.append(1)
        else:
            res.append(0)

    return res.count(1)


def get_incorrectly_classified_2(vector, ytest):
    res = []
    for i in range(0, len(vector)):
        if vector[i] == ytest[i]:
            res.append(1)
        else:
            res.append(0)

    return res.count(0)
