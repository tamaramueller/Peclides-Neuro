from __future__ import division
import pandas as pd


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


def get_index_min(l):
    return l.index(min(l))


def does_rule_contain_feature(feature, rule):
    for i in range(0, len(rule)-1):
        if rule[i][0] == feature:
            return True
    return False


def does_rule_apply2(rule, row):
    for i in range(0, len(rule) - 1):
        if rule[i][1] == 'l':
            if row[int(rule[i][0])] > rule[i][2]:
                return False
        else:
            if row[int(rule[i][0])] <= rule[i][2]:
                return False

    return True


def apply_one_rule_to_one_row(rule, row):
    if rule[len(rule) - 1][0] > rule[len(rule) - 1][1]:
        pred = 0
    else:
        pred = 1

    for i in range(0, len(rule) - 1):
        if rule[i][1] == 'l':
            if row[rule[i][0]] > rule[i][2]:
                return 1 - pred
        else:
            if row[rule[i][0]] <= rule[i][2]:
                return 1 - pred

        if i == len(rule) - 2:
            return pred


def apply_ruleset_one_row(ruleset, row):
    res = []
    for rule in ruleset:
        res.append(apply_one_rule_to_one_row(rule=rule, row=row))

    if res.count(0) > res.count(1):
        return 0
    else:
        return 1


def apply_ruleset_one_row_new(ruleset, row):
    res = []
    for rule in ruleset:
        if does_rule_apply2(row=row, rule=rule):
            if rule[len(rule) - 1][0] > rule[len(rule) - 1][1]:
                res.append(0)
            else:
                res.append(1)

    if res.count(0) > res.count(1):
        return 0
    else:
        return 1


def get_specificity(reslist, truevals):
    true_negatives = 0
    false_positives = 0
    if len(reslist) != len(truevals):
        return -1
    for i in range(0, len(reslist)):
        if (truevals[i] == 0) & (reslist[i] == 0):
            true_negatives = true_negatives + 1
        if (truevals[i] == 0) & (reslist[i] == 1):
            false_positives = false_positives + 1
    if true_negatives == 0:
        return 0
    if true_negatives+false_positives == 0:
        return 0
    return true_negatives / (true_negatives+false_positives)


def get_sensitivity(reslist, truevals):
    true_positives = 0
    false_negatives = 0
    if len(reslist) != len(truevals):
        return -1
    for i in range(0, len(reslist)):
        if (truevals[i] == 1) & (reslist[i] == 1):
            true_positives = true_positives + 1
        if (reslist[i] == 0) & (truevals[i] == 1):
            false_negatives = false_negatives + 1
    if true_positives == 0:
        return 0
    if true_positives+false_negatives == 0:
        return 0
    return true_positives / (true_positives+false_negatives)


def get_number_feat_in_rules(ruleset, features):
    vec_number_rules_containing_feature = []

    for i in features:
        vec_number_rules_containing_feature.append(rule_contains_feature(feature=i, ruleset=ruleset).count(1))

    return vec_number_rules_containing_feature


def rule_contains_feature(feature, ruleset):
    res = []
    for x in ruleset:
        found = 0
        for i in range(0, len(x) - 1):
            if (x[i][0] == feature):
                found = 1
                break
        if (found == 1):
            res.append(1)
        else:
            res.append(0)

    return res


def create_dataframe(name):
    datafr = pd.read_csv(name)
    return datafr


def create_separate_files_status(name):
    df = create_dataframe(name)
    l = []

    for i in range(1, 51):

        if (i < 10):
            st = "S0" + str(i)
        else:
            st = "S" + str(i)

        for j in range(0, 195):
            if (st in df['name'][j]):
                l.append(df['Status'][j])

    return l;