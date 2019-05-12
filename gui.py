from __future__ import division
from Tkinter import *
import tkMessageBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np


from reducer import Reducer
import helpers


red_ruleset = []
new_ruleset = []
DEBUG = False


class Gui:

    def __init__(self, rule_list, random_forest):
        self.random_forest = random_forest
        self.reducer = Reducer(rule_list, self.random_forest)

        # self.red_ruleset = rule_list
        # self.new_ruleset = rule_list

    @staticmethod
    def print_rule(rule, feature_names):
        ret = 'if '
        for i in range(0, len(rule) - 1):
            if rule[i][1] == 'l':
                lower_greater = '<='
            else:
                lower_greater = '>'
            ret += feature_names[rule[i][0]] + " " + lower_greater + " " + str(rule[i][2])
            if i < len(rule) - 2:
                ret += ' and '
            else:
                ret += ' then '

        if rule[len(rule) - 1][0] > rule[len(rule) - 1][1]:
            ret += '\nHEALTHY!'
        else:
            ret += '\nDISEASED!'

        return ret

    @staticmethod
    def print_all_rules(ruleset, feature_names):
        ret = ''
        for rule in ruleset:
            ret += Gui.print_rule(rule, feature_names)
            ret += ' \n '

        return ret

    # implementation of the GUI
    # feature_names: list of strings with all feature names
    # ruleset: array of array of arrays with all rules
    # X_train: dataframe with all data samples of training set
    # y_train: ground truth of X_train as array
    # X_test: dataframe with all data samples of test set
    # y_test: ground truth of X_test as array
    def window(self, feature_names, data_set_name, ruleset, X_train, y_train, X_test, y_test):
        global red_ruleset
        global new_ruleset

        red_ruleset = ruleset
        new_ruleset = ruleset

        feature_info = "Please name your favourite features. Rules containing them will be less likely to be deleted " \
                       "from the rule set. You can name as many as you want. The order matters: the first feature is " \
                       "treated as the most preferred one. Please separate the features with a comma. An example would " \
                       "be: \n \t 1,2,3"
        perc_info = "Pleas name the percentage of the size of the original rule set, you would like the reduced rule set " \
                    "to have. Please only type in the number, without the percent sign. An example would be: \n \t 30"

        # eliminate useless queries within a rule
        def first_reduction():
            global red_ruleset
            global new_ruleset

            red_ruleset = self.reducer.reduce_rules()
            red1_label.config(text="new rule size: " + str(len(red_ruleset)))

        # reduce the rule set based on given percentage and preferred features
        def reduce_action():
            global new_ruleset
            global red_ruleset

            features = eingabefeld.get()
            percentage = entrytext.get()

            if features == "":
                features = []
            else:
                features = helpers.string_to_int_list(features)

            if percentage == "":
                reduce_label.config(text="no percentage set")
            else:
                numtoelim = int((1 - (int(percentage) / 100)) * len(red_ruleset))
                new_ruleset = self.reducer.eliminate_weakest_rules_2(favourite_features=features, k=4, numtoelim=numtoelim,
                                                        ruleset=red_ruleset, xtrain=X_train, ytrain=y_train)
                vector_pred = self.random_forest.apply_ruleset_get_vector_new(ruleset=new_ruleset, xtest=X_test)

                if DEBUG:
                    print("gui: vector pred len: %s" % len(vector_pred))
                    print("gui: y_test len: %s" % len(y_test))

                acc = self.random_forest.get_accuracy_of_ruleset_new(ruleset=new_ruleset, xtest=X_test, ytest=y_test)

                spec = helpers.get_specificity(reslist=vector_pred, truevals=y_test)
                if DEBUG:
                    print("gui: spec: %s" % spec)
                sens = helpers.get_sensitivity(reslist=vector_pred, truevals=y_test)

                reduce_label.config(text="New Rule Size:  " + str(len(new_ruleset)))
                acc_label.config(
                    text="Accuracy: " + str(acc) + ", Sensitivity: " + str(sens) + ", Specificity: " + str(spec))

        def predict_action():
            global new_ruleset
            global red_ruleset

            f0_text = e_f0.get()
            f1_text = e_f1.get()
            f2_text = e_f2.get()
            f3_text = e_f3.get()
            f4_text = e_f4.get()
            f5_text = e_f5.get()
            f6_text = e_f6.get()
            f7_text = e_f7.get()
            f8_text = e_f8.get()
            f9_text = e_f9.get()
            f10_text = e_f10.get()
            f11_text = e_f11.get()
            f12_text = e_f12.get()
            f13_text = e_f13.get()
            f14_text = e_f14.get()
            f15_text = e_f15.get()
            f16_text = e_f16.get()
            f17_text = e_f17.get()
            f18_text = e_f18.get()
            f19_text = e_f19.get()
            f20_text = e_f20.get()
            f21_text = e_f21.get()

            if ((f0_text == "") | (f1_text == "") | (f2_text == "") | (f3_text == "") | (f4_text == "") |
                    (f5_text == "") | (f6_text == "") | (f7_text == "") | (f8_text == "") | (f9_text == "") |
                    (f10_text == "") | (f11_text == "") | (f12_text == "") | (f13_text == "") | (f14_text == "") |
                    (f15_text == "") | (f16_text == "") | (f17_text == "") | (f18_text == "") | (f19_text == "") |
                    (f20_text == "") | (f21_text == "")):
                predict_label.config(text="not all features set")
            else:
                vec = [float(f0_text), float(f1_text), float(f2_text), float(f3_text), float(f4_text),
                       float(f5_text), float(f6_text), float(f7_text), float(f8_text), float(f9_text),
                       float(f10_text), float(f11_text), float(f12_text), float(f13_text), float(f14_text),
                       float(f15_text), float(f16_text), float(f17_text), float(f18_text), float(f19_text),
                       float(f20_text), float(f21_text)]

                df = pd.DataFrame([vec], columns=feature_names)
                pred = self.random_forest.apply_ruleset_get_vector_new(ruleset=new_ruleset, xtest=df)

                if pred[0] == 0:
                    string = "HEALTHY!"
                else:
                    string = "ALZHEIMERS DISEASE"

                predict_label.config(text="Prediction:  " + string + "!")

        def message_features():
            tkMessageBox.showinfo("Favourite Features", feature_info)

        def message_percentage():
            tkMessageBox.showinfo("Percentage", perc_info)

        def print_rules_():
            win = Toplevel(fenster)
            win.title("All Rules in Reduced Rule Set")
            scroll = Scrollbar(win)
            # scroll.pack(side = RIGHT, fill = Y)
            scroll.grid(row=0, column=1, sticky=N + S)

            txt = Text(win, wrap=WORD, yscrollcommand=scroll.set, xscrollcommand=scroll.set)
            txt.grid(row=0, column=0, sticky=N + S + E + W)
            # txt.insert(INSERT, build_string_ruleset(ruleset=self.new_ruleset, featurenames=feature_names))
            txt.insert(INSERT, Gui.print_all_rules(new_ruleset, feature_names))
            # txt.insert(INSERT, "TEST")

            scroll.config(command=txt.yview)

        def bar_chart_orig_rules():
            global new_ruleset
            global red_ruleset

            wind = Toplevel(fenster)
            wind.title("Number of rules containing respective features in original rule set")

            f = Figure(figsize=(5, 4), dpi=100)
            ax = f.add_subplot(111)

            data = helpers.get_number_feat_in_rules(ruleset=red_ruleset, features=range(0, 22))

            ind = np.arange(22)
            width = .5

            rects1 = ax.bar(ind, data, width)

            canvas = FigureCanvasTkAgg(f, master=wind)
            canvas.draw()
            canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        def bar_chart_red_rules():
            global new_ruleset
            global red_ruleset

            wind = Toplevel(fenster)
            wind.title("Number of rules containing respective features in reduced rule set")

            f = Figure(figsize=(5, 4), dpi=100)
            ax = f.add_subplot(111)

            data = helpers.get_number_feat_in_rules(ruleset=new_ruleset, features=range(0, 22))

            ind = np.arange(22)  # the x locations for the groups
            width = .5

            rects1 = ax.bar(ind, data, width)

            canvas = FigureCanvasTkAgg(f, master=wind)
            canvas.draw()
            canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        # creating main window
        fenster = Tk()
        fenster.title("Decision Support")

        # information labels
        dataset = Label(fenster, text=data_set_name)
        numrules = Label(fenster, text="Number of Rules: " + str(len(ruleset)))
        feat_label = Label(fenster, text="Favourite Features (optional) ")
        perc_label = Label(fenster, text="Percentage")
        label_f0 = Label(fenster, text=feature_names[0])
        label_f1 = Label(fenster, text=feature_names[1])
        label_f2 = Label(fenster, text=feature_names[2])
        label_f3 = Label(fenster, text=feature_names[3])
        label_f4 = Label(fenster, text=feature_names[4])
        label_f5 = Label(fenster, text=feature_names[5])
        label_f6 = Label(fenster, text=feature_names[6])
        label_f7 = Label(fenster, text=feature_names[7])
        label_f8 = Label(fenster, text=feature_names[8])
        label_f9 = Label(fenster, text=feature_names[9])
        label_f10 = Label(fenster, text=feature_names[10])
        label_f11 = Label(fenster, text=feature_names[11])
        label_f12 = Label(fenster, text=feature_names[12])
        label_f13 = Label(fenster, text=feature_names[13])
        label_f14 = Label(fenster, text=feature_names[14])
        label_f15 = Label(fenster, text=feature_names[15])
        label_f16 = Label(fenster, text=feature_names[16])
        label_f17 = Label(fenster, text=feature_names[17])
        label_f18 = Label(fenster, text=feature_names[18])
        label_f19 = Label(fenster, text=feature_names[19])
        label_f20 = Label(fenster, text=feature_names[20])
        label_f21 = Label(fenster, text=feature_names[21])

        red1_label = Label(fenster)
        reduce_label = Label(fenster)
        predict_label = Label(fenster)
        acc_label = Label(fenster)

        # Here the user can enter something
        eingabefeld = Entry(fenster, bd=5, width=40)
        entrytext = Entry(fenster, bd=5, width=40)
        e_f0 = Entry(fenster, bd=5, width=8)
        e_f1 = Entry(fenster, bd=5, width=8)
        e_f2 = Entry(fenster, bd=5, width=8)
        e_f3 = Entry(fenster, bd=5, width=8)
        e_f4 = Entry(fenster, bd=5, width=8)
        e_f5 = Entry(fenster, bd=5, width=8)
        e_f6 = Entry(fenster, bd=5, width=8)
        e_f7 = Entry(fenster, bd=5, width=8)
        e_f8 = Entry(fenster, bd=5, width=8)
        e_f9 = Entry(fenster, bd=5, width=8)
        e_f10 = Entry(fenster, bd=5, width=8)
        e_f11 = Entry(fenster, bd=5, width=8)
        e_f12 = Entry(fenster, bd=5, width=8)
        e_f13 = Entry(fenster, bd=5, width=8)
        e_f14 = Entry(fenster, bd=5, width=8)
        e_f15 = Entry(fenster, bd=5, width=8)
        e_f16 = Entry(fenster, bd=5, width=8)
        e_f17 = Entry(fenster, bd=5, width=8)
        e_f18 = Entry(fenster, bd=5, width=8)
        e_f19 = Entry(fenster, bd=5, width=8)
        e_f20 = Entry(fenster, bd=5, width=8)
        e_f21 = Entry(fenster, bd=5, width=8)

        reduce_rule_set_button = Button(fenster, text="Reduce Rule Set", command=reduce_action)
        predict_button = Button(fenster, text="Predict", command=predict_action)
        red1_button = Button(fenster, text="First Reduction", command=first_reduction)

        bar_chart_orig_button = Button(fenster, text="Show Features in Original Rule Set", command=bar_chart_orig_rules)
        bar_chart_red_button = Button(fenster, text="Show Features in Reduced Rule Set", command=bar_chart_red_rules)

        info_feat_button = Button(fenster, text="more info", command=message_features)
        info_perc_button = Button(fenster, text="more info", command=message_percentage)
        info_rules_button = Button(fenster, text="Print Rules", command=print_rules_)

        dataset.grid(row=0, column=0, columnspan=5)
        numrules.grid(row=0, column=6, columnspan=5)

        feat_label.grid(row=4, column=2, columnspan=3)
        perc_label.grid(row=5, column=2, columnspan=3)
        eingabefeld.grid(row=4, column=4, columnspan=5)
        reduce_rule_set_button.grid(row=6, column=1, columnspan=9)
        entrytext.grid(row=5, column=4, columnspan=5)
        predict_button.grid(row=12, column=1, columnspan=9)
        info_rules_button.grid(row=15, column=1, columnspan=9)
        # exit_button.grid(row = 4, column = 1)
        reduce_label.grid(row=7, column=0, columnspan=3)
        predict_label.grid(row=13, column=1, columnspan=9)
        acc_label.grid(row=7, column=3, columnspan=8)

        red1_button.grid(row=2, column=1, columnspan=9)
        red1_label.grid(row=3, column=1, columnspan=9)

        bar_chart_orig_button.grid(row=17, column=0, columnspan=5)
        bar_chart_red_button.grid(row=17, column=6, columnspan=5)

        info_feat_button.grid(row=4, column=9)
        info_perc_button.grid(row=5, column=9)

        label_f0.grid(row=8, column=0)
        label_f1.grid(row=8, column=1)
        label_f2.grid(row=8, column=2)
        label_f3.grid(row=8, column=3)
        label_f4.grid(row=8, column=4)
        label_f5.grid(row=8, column=5)
        label_f6.grid(row=8, column=6)
        label_f7.grid(row=8, column=7)
        label_f8.grid(row=8, column=8)
        label_f9.grid(row=8, column=9)
        label_f10.grid(row=8, column=10)
        label_f11.grid(row=10, column=0)
        label_f12.grid(row=10, column=1)
        label_f13.grid(row=10, column=2)
        label_f14.grid(row=10, column=3)
        label_f15.grid(row=10, column=4)
        label_f16.grid(row=10, column=5)
        label_f17.grid(row=10, column=6)
        label_f18.grid(row=10, column=7)
        label_f19.grid(row=10, column=8)
        label_f20.grid(row=10, column=9)
        label_f21.grid(row=10, column=10)

        e_f0.grid(row=9, column=0)
        e_f1.grid(row=9, column=1)
        e_f2.grid(row=9, column=2)
        e_f3.grid(row=9, column=3)
        e_f4.grid(row=9, column=4)
        e_f5.grid(row=9, column=5)
        e_f6.grid(row=9, column=6)
        e_f7.grid(row=9, column=7)
        e_f8.grid(row=9, column=8)
        e_f9.grid(row=9, column=9)
        e_f10.grid(row=9, column=10)
        e_f11.grid(row=11, column=0)
        e_f12.grid(row=11, column=1)
        e_f13.grid(row=11, column=2)
        e_f14.grid(row=11, column=3)
        e_f15.grid(row=11, column=4)
        e_f16.grid(row=11, column=5)
        e_f17.grid(row=11, column=6)
        e_f18.grid(row=11, column=7)
        e_f19.grid(row=11, column=8)
        e_f20.grid(row=11, column=9)
        e_f21.grid(row=11, column=10)

        fenster.mainloop()
