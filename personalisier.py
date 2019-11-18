import helpers


class Personaliser:

    def __init__(self, random_forest):
        self.random_forest = random_forest

    # calculate rule score 2 from paper (considering length of rule)
    # with new apply method
    def get_rule_score2_new_apply(self, ruleset, xtrain, ytrain, k):
        res = []

        for x in ruleset:
            # get vector with categorisations of each rule
            vec_cat = self.random_forest.apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)

            # get correct and incorrect classificatoins
            vec_compare_ytrain = helpers.compare_two_lists(list1=vec_cat, list2=ytrain)

            score1 = ((helpers.get_correctly_classified_2(vector=vec_compare_ytrain,
                                                  ytest=ytrain) - helpers.get_incorrectly_classified_2(
                vector=vec_compare_ytrain, ytest=ytrain)) / (helpers.get_correctly_classified_2(vector=vec_compare_ytrain,
                                                                                        ytest=ytrain) + helpers.get_incorrectly_classified_2(
                vector=vec_compare_ytrain, ytest=ytrain))) + (
                                 (helpers.get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)) / (
                                     helpers.get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain) + k))
            score2 = score1 + (helpers.get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain) / len(x))
            res.append(score2)

        return res

    def get_rule_score2_pers_set_new_apply(self, ruleset, xtrain, ytrain, k, favourite_features):
        res = []

        for x in ruleset:
            # get vector with categorisations of each rule
            vec_cat = self.random_forest.apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)

            # get correct and incorrect classifications
            # vec_compare_ytrain = compare_two_lists(list1=vec_cat, list2=ytrain)

            score1 = ((helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain) - helpers.get_incorrectly_classified_2(
                vector=vec_cat, ytest=ytrain)) / (helpers.get_correctly_classified_2(vector=vec_cat,
                                                                             ytest=ytrain) + helpers.get_incorrectly_classified_2(
                vector=vec_cat, ytest=ytrain))) + ((helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain)) / (
                        helpers.get_incorrectly_classified_2(vector=vec_cat, ytest=ytrain) + k))
            score2 = score1 + (helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain) / len(x))
            # add value 10 if rule contains favourite feature
            addition = 0
            # add a new value depending on how important the feature is that is included
            # in the rule
            for i in range(0, len(favourite_features)):
                if (helpers.does_rule_contain_feature(feature=favourite_features[i], rule=x)):
                    addition = addition + ((1 / (i + 1)) * 40)

            score_pers = score2 + addition
            res.append(score_pers)

        return res

    def get_rule_score_only_performance(self, ruleset, xtrain, ytrain, k, favourite_features):
        res = []

        for x in ruleset:
            # get vector with categorisations of each rule
            vec_cat = self.random_forest.apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)

            # get correct and incorrect classifications
            # vec_compare_ytrain = compare_two_lists(list1=vec_cat, list2=ytrain)

            score1 = ((helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain) - helpers.get_incorrectly_classified_2(
                vector=vec_cat, ytest=ytrain)) /
                      (helpers.get_correctly_classified_2(vector=vec_cat,ytest=ytrain) + helpers.get_incorrectly_classified_2(
                vector=vec_cat, ytest=ytrain))) + ((helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain)) /
                                                   (helpers.get_incorrectly_classified_2(vector=vec_cat, ytest=ytrain) + k))
            score2 = score1  # + (helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain) / len(x))
            # add value 10 if rule contains favourite feature
            addition = 0
            # add a new value depending on how important the feature is that is included
            # in the rule
            for i in range(0, len(favourite_features)):
                if (helpers.does_rule_contain_feature(feature=favourite_features[i], rule=x)):
                    addition = addition + ((1 / (i + 1)) * 40)

            score_pers = score2 + addition
            res.append(score_pers)

        return res

    def get_rule_score_only_accruacy(self, ruleset, xtrain, ytrain, k, favourite_features):
        res = []

        for x in ruleset:
            # get vector with categorisations of each rule
            vec_cat = self.random_forest.apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)

            # get correct and incorrect classifications
            # vec_compare_ytrain = compare_two_lists(list1=vec_cat, list2=ytrain)

            score1 = ((helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain)) / #- helpers.get_incorrectly_classified_2(
                      (helpers.get_correctly_classified_2(vector=vec_cat,ytest=ytrain) + helpers.get_incorrectly_classified_2(
                vector=vec_cat, ytest=ytrain))) #+ ((helpers.get_correctly_classified_2(vector=vec_cat, ytest=ytrain)) /
                                              #     (helpers.get_incorrectly_classified_2(vector=vec_cat, ytest=ytrain) + k))
            # add value 10 if rule contains favourite feature
            addition = 0
            # add a new value depending on how important the feature is that is included
            # in the rule
            for i in range(0, len(favourite_features)):
                if (helpers.does_rule_contain_feature(feature=favourite_features[i], rule=x)):
                    addition = addition + ((1 / (i + 1)) * 40)

            score_pers = score1 + addition
            res.append(score_pers)

        return res
