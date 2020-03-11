from __future__ import division
import copy

import statistics

import helpers
import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


class RandomForest:

    def __init__(self, x_test, y_test, x_train, y_train, criterion="gini"):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.rf = self.create_random_forest(n_estimators=30, max_depth=15, crit=criterion, x_train=x_train, y_train=y_train,
                                  random_state=42)
        self.orig_ruleset = None
        self.red_ruleset = None
        self.score = self.rf.score(self.x_test, self.y_test)

    def create_random_forest(self, n_estimators, crit, max_depth, x_train, y_train, random_state):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=crit,
                                    random_state=random_state)
        rf.fit(x_train, y_train)
        return rf

    # def apply_rule_get_vector2(self, rule):
    #     # returns classification vector of one rule
    #     reslist = []
    #
    #     for row in self.x_test.values.tolist():
    #         reslist.append(helpers.apply_ruleset_one_row(ruleset=[rule], row=row))
    #
    #     return reslist

    def apply_rule_get_vector2(self, XTest, ytest, rule):
        # returns classification vector of one rule
        reslist = []

        for row in XTest.values.tolist():
            reslist.append(helpers.apply_ruleset_one_row(ruleset=[rule], row=row))

        return reslist

    def get_accuracy_of_ruleset_new(self, ruleset, xtest, ytest):
        vec = self.apply_ruleset_get_vector_new(ruleset=ruleset, xtest=xtest)

        if len(vec) != len(ytest):
            print("not the same length")
            return -1

        correct = helpers.get_correctly_classified_2(vector=vec, ytest=ytest)

        return correct / len(vec)

    def get_mse_of_ruleset_new(self, ruleset, xtest, ytest):
        vec = self.apply_ruleset_get_vector_new(ruleset=ruleset, xtest=xtest)

        return mean_squared_error(ytest, vec)

    def calculate_model_performance_parameter2(self, ruleset, xtest, ytest):
        vec = self.apply_ruleset_get_vector_new(ruleset=ruleset, xtest=xtest)

        return 1 - (mean_squared_error(ytest, vec) / statistics.variance(ytest)) \
               + (((np.mean(ytest) - np.mean(vec)) ** 2) / statistics.variance(ytest))

    def apply_ruleset_get_vector_new(self, ruleset, xtest):
        res = []
        for row in xtest.values.tolist():
            res.append(helpers.apply_ruleset_one_row_new(ruleset=ruleset, row=row))
        return res

    def get_all_rules(self, forest):
        li = []
        for i in forest:
            li = li + self.get_branch(i)

        return li

    def get_branch(self, t):
        l = []
        ll = []
        tt = t.tree_
        node = 0
        depths = self.get_node_depths(t)
        # while (tt.children_left[node] != _tree.TREE_LEAF):
        while (node < tt.node_count):
            while (tt.feature[node] != _tree.TREE_UNDEFINED):
                l.append([tt.feature[node], "l", tt.threshold[node]])
                # print("feature: {}".format(tt.feature[node]))
                # print("left child: {}".format(tt.children_left[node]))
                node = node + 1
            # print("node: {}".format(tt.value[9]))
            l.append([tt.value[node][0][0], tt.value[node][0][1], 0])

            # l.append([tt.value[node],0,0])
            ll.append(copy.deepcopy(l))
            # print("value: {}".format(tt.value[node]))
            # print("value 1: {}".format(tt.value[node][0][0]))
            # print("value 2: {}".format(tt.value[node][0][1]))
            if (node != tt.node_count - 1):
                for i in range(0, abs(depths[node] - depths[node + 1])):
                    # print("got ya")
                    # print("l before: {}".format(l))
                    del l[-1]
                    # print("l after: {}".format(l))

            del l[-1]
            # print("test: {}".format(l[len(l)-1]))
            # print("l before: {}".format(l))

            # tmp = copy.copy(l)
            if (len(l) >= 1):
                l[len(l) - 1][1] = 'g'
            # l = copy.copy(tmp)
            # print("l after: {}".format(l))

            # print("l: {}".format(l))
            # print("node count {}".format(tt.node_count))

            node = node + 1
        # print(len(depths))
        # print(abs(depths[node-2] - depths[node-1]))
        # print("right of 7: {}".format(tt.children_right[7]))
        # print("depths: {}".format(depths))
        return ll

    def get_node_depths(self, t):

        tt = t.tree_

        def get_node_depths_(current_node, current_depth, l, r, depths):
            depths += [current_depth]
            if l[current_node] != -1 and r[current_node] != -1:
                get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
                get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

        depths = []
        get_node_depths_(0, 0, tt.children_left, tt.children_right, depths)
        return np.array(depths)
