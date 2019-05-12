from sklearn.model_selection import train_test_split

from random_forest import RandomForest
from gui import Gui
import config
import helpers


red_ruleset = []
new_ruleset = []

feature_names = config.feature_names

if __name__ == '__main__':
    groundTruth = helpers.create_separate_files_status("data.csv")
    x_train, x_test, y_train, y_test = train_test_split(helpers.create_dataframe("dataWoName.csv"), groundTruth, test_size=0.33,
                                                        random_state=42)

    rf = RandomForest(x_test, y_test, x_train, y_train)
    rules_rf = rf.get_all_rules(forest=rf.rf)
    gui = Gui(rules_rf, rf)

    gui.window(feature_names, "Speech Data Set", rules_rf, x_train, y_train, x_test, y_test)
