import helpers


class Personaliser:

    def __init__(self, random_forest):
        self.random_forest = random_forest

    # calculate rule score 2 from paper (considering length of rule)
    # with new apply method
    def get_rule_score2_newApply(self, ruleset, xtrain, ytrain, k):
        res = []

        for x in ruleset:
            # get vector with categorisations of each rule
            vec_cat = apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)

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