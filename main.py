from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

red_ruleset = []
new_ruleset = []


if __name__ == '__main__':
    window(feature_names, "Speech Data Set", rules_rf, X_train, y_train, X_test, y_test)
