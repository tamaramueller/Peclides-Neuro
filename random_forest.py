


class RandomForest():

    def __init__(self):
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None

    def apply_rule_get_vector2(self, rule):
        # returns classification vector of one rule
        reslist = []

        for row in self.x_test.values.tolist():
            reslist.append(self.apply_ruleset_one_row(ruleset=[rule], row=row))

        return

    def apply_ruleset_one_row2(self, ruleset, row):
        res = []
        for rule in ruleset:
            res.append(apply_one_rule_to_one_row(rule=rule, row=row))

        # print(res)
        # return res
        # return (sum(res) / len(res))
        if (res.count(0) > res.count(1)):
            return 0
        else:
            return 1

