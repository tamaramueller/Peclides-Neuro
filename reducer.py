import copy

from personalisier import Personaliser

class Reducer:

    def __init__(self, rule_list):
        self.rule_list = rule_list
        self.len = len(self.rule_list)
        self.personaliser = Personaliser()

    def reduce_rules(self):
        to_del = []
        for i in range(0, self.len):
            for k in range(0, len(self.rule_list[i])):

                for j in range(1, (len(self.rule_list[i]) - k)):
                    if ((self.rule_list[i][k][0] == self.rule_list[i][k + j][0]) &
                            (self.rule_list[i][k][1] == self.rule_list[i][k + j][1])):
                        to_del.append([i, k])

        l = copy.deepcopy(self.rule_list)

        for index in sorted(to_del, reverse=True):
            del l[index[0]][index[1]]

        return l

    def eliminate_weakest_rules_2(self, favourite_features, numtoelim, ruleset, xtrain, ytrain, k):

        if favourite_features == []:
            rulescores = get_rule_score2_newApply(ruleset=ruleset, xtrain=xtrain, ytrain=ytrain, k=k)
        else:
            rulescores = get_rule_score2_pers_set_newApply(ruleset=ruleset, xtrain=xtrain, ytrain=ytrain, k=k,
                                                           favourite_features=favourite_features)

        rules = copy.deepcopy(ruleset)

        for i in range(0, numtoelim):
            index = get_index_min(rulescores)
            # print(index)
            del rules[index]
            del rulescores[index]

        return rules