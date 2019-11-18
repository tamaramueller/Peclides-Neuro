import copy
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

from personalisier import Personaliser
import helpers
from random_forest import RandomForest


class Reducer:

    def __init__(self, rule_list, random_forest):
        self.rule_list = rule_list
        self.len = len(self.rule_list)
        self.personaliser = Personaliser(random_forest)

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

        rule_scores = self.personaliser.get_rule_score2_pers_set_new_apply(ruleset=ruleset, xtrain=xtrain, ytrain=ytrain,
                                                                           k=k, favourite_features=favourite_features)

        rules = copy.deepcopy(ruleset)

        for i in range(0, numtoelim):
            index = helpers.get_index_min(rule_scores)
            del rules[index]
            del rule_scores[index]

        return rules

    def eliminate_weakest_rules_score1(self, favourite_features, numtoelim, ruleset, xtrain, ytrain, k):

        rule_scores = self.personaliser.get_rule_score_only_performance(ruleset=ruleset, xtrain=xtrain, ytrain=ytrain,
                                                                           k=k, favourite_features=favourite_features)

        rules = copy.deepcopy(ruleset)

        for i in range(0, numtoelim):
            index = helpers.get_index_min(rule_scores)
            del rules[index]
            del rule_scores[index]

        return rules

    def eliminate_weakest_rules_acc_only(self, favourite_features, numtoelim, ruleset, xtrain, ytrain, k):

        rule_scores = self.personaliser.get_rule_score_only_accruacy(ruleset=ruleset, xtrain=xtrain, ytrain=ytrain,
                                                                           k=k, favourite_features=favourite_features)

        rules = copy.deepcopy(ruleset)

        for i in range(0, numtoelim):
            index = helpers.get_index_min(rule_scores)
            del rules[index]
            del rule_scores[index]

        return rules


if __name__ == '__main__':

    groundTruth = helpers.create_separate_files_status("data.csv")
    percentage = np.arange(1, 100, 1)

    kf = KFold(n_splits=2)
    data = helpers.create_dataframe("dataWoName.csv")

    ground_truth = np.array(groundTruth)
    all_accs = []
    all_mse = []

    for train_index, test_index in kf.split(X=data, y=ground_truth):
        # print(train_index, test_index)
        x_train = data.iloc[train_index]
        y_train = ground_truth[train_index]
        x_test = data.iloc[test_index]
        y_test = ground_truth[test_index]

        print(len(x_train), len(y_train), len(x_test), len(y_test))

        rf = RandomForest(x_test, y_test, x_train, y_train)
        rules_rf = rf.get_all_rules(forest=rf.rf)
        reducer = Reducer(rules_rf, rf)
        red_ruleset = reducer.reduce_rules()
        numtoelim = [int((1 - (int(x) / 100)) * len(red_ruleset)) for x in percentage]

        print(len(red_ruleset))

        # new_ruleset = reducer.eliminate_weakest_rules_2(favourite_features=[], k=4, numtoelim=30, ruleset=red_ruleset, xtrain=x_train, ytrain=y_train)
        # acc = rf.get_accuracy_of_ruleset_new(ruleset=new_ruleset, xtest=x_test, ytest=y_test)
        new_ruleset = [
            reducer.eliminate_weakest_rules_2(favourite_features=[], k=4, numtoelim=x, ruleset=red_ruleset,
                                              xtrain=x_train, ytrain=y_train) for x in numtoelim]
        acc = [rf.get_accuracy_of_ruleset_new(ruleset=x, xtest=x_test, ytest=y_test) for x in new_ruleset]
        mse = [rf.get_mse_of_ruleset_new(ruleset=x, xtest=x_test, ytest=y_test) for x in new_ruleset]

        # sns.lineplot(percentage, acc)
        # plt.show()
        all_accs.append(acc)
        all_mse.append(mse)

    dataframe = pd.DataFrame()
    dataframe["percentage_of_rule_set_kept"] = ['fold1', 'fold2']
    dataframe["accuracy"] = all_accs

    #ax = sns.lineplot(x="percentage_of_rule_set_kept", y="accuracy", data=dataframe)

