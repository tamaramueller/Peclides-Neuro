#! /usr/bin/python

from sklearn.tree import _tree
from sklearn import tree
import graphviz
from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import csv
import pandas as pd
from statistics import mean
from statistics import variance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from statistics import median
import numpy as np
import sys, argparse
import scipy
import matplotlib as plt
from csv import reader
import mglearn
import copy
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import os
import subprocess
from __future__ import division
import math
import os, sys
import subprocess
import sklearn
from sklearn.tree import _tree
# for MR images
from nipy import load_image
from nilearn import plotting
from nilearn import image
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF




# implementation of the GUI
# feature_names: list of strings with all feature names
# ruleset: array of array of arrays with all rules
# X_train: dataframe with all data samples of training set
# y_train: ground truth of X_train as array
# X_test: dataframe with all data samples of test set
# y_test: ground truth of X_test as array 
def window(feature_names, data_set_name, ruleset, X_train, y_train, X_test, y_test):
    global red_ruleset
    global new_ruleset
    
    red_ruleset=ruleset
    new_ruleset=ruleset

    feature_info = "Please name your favourite features. Rules containing them will be less likely to be deleted from the rule set. You can name as many as you want. The order matters: the first feature is treated as the most preferred one. Please separate the features with a comma. An example would be: \n \t 1,2,3"
    perc_info = "Pleas name the percentage of the size of the original rule set, you would like the reduced rule set to have. Please only type in the number, without the percent sign. An example would be: \n \t 30"
    
    # eliminate useless queries within a rule
    def first_reduction():
        red_ruleset = reduce_rules(rulelist=ruleset)
        red1_label.config(text="new rule size: " + str(len(red_ruleset)))

    # reduce the rule set based on given percentage and preferred features
    def reduce_action():
        global new_ruleset
        global red_ruleset
        
        features = eingabefeld.get()
        percentage = entrytext.get()
        
        if(features == ""):
            featues = []
        else: 
            features = string_to_intlist(features)
        
        if ((percentage=="")):
            reduce_label.config(text="no percentage set")
        else:
            size=0
            acc=0
            spec = 0
            sens= 0

            numtoelim = int((1-(int(percentage)/100)) * len(red_ruleset))
            new_ruleset = eliminate_weakest_rules_2(favourite_features=features, k=4, numtoelim=numtoelim, ruleset=red_ruleset, xtrain=X_train, ytrain=y_train)
            vector_pred = apply_ruleset_get_vector_new(ruleset=new_ruleset, xtest=X_test)
            
            acc = get_accuracy_of_ruleset_new(ruleset=new_ruleset, xtest=X_test, ytest=y_test)
            
            spec = get_specificity(reslist=vector_pred ,truevals=y_test)
            sens = get_sensitivity(reslist=vector_pred, truevals=y_test)
            
            reduce_label.config(text="New Rule Size:  " + str(len(new_ruleset)))
            acc_label.config(text="Accuracy: " + str(acc) + ", Sensitivity: " + str(sens) + ", Specificity: " + str(spec))


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
        
        if ((f0_text == "") | (f1_text=="") | (f2_text=="") | (f3_text=="") | (f4_text=="") | 
           (f5_text=="") | (f6_text=="") | (f7_text=="") | (f8_text=="") | (f9_text=="") | 
           (f10_text=="") | (f11_text=="") | (f12_text=="") | (f13_text=="") | (f14_text=="") | 
           (f15_text=="") | (f16_text=="") | (f17_text=="") | (f18_text=="") | (f19_text=="") | 
           (f20_text=="") | (f21_text=="") ):
            predict_label.config(text="not all features set")
        else:
            vec = [float(f0_text), float(f1_text), float(f2_text), float(f3_text), float(f4_text),
                  float(f5_text), float(f6_text), float(f7_text), float(f8_text), float(f9_text),
                  float(f10_text), float(f11_text), float(f12_text), float(f13_text), float(f14_text),
                  float(f15_text), float(f16_text), float(f17_text) ,float(f18_text), float(f19_text), 
                  float(f20_text), float(f21_text)]
            
            df = pd.DataFrame([vec], columns=feature_names)
            pred = apply_ruleset_get_vector_new(ruleset=new_ruleset, xtest=df)
            
            if(pred[0] == 0):
                string = "HEALTHY!"
            else:
                string = "ALZHEIMERS DISEASE"
            
            predict_label.config(text="Prediction:  " + string + "!")

            
    def message_features():
        tkMessageBox.showinfo("Favourite Features", feature_info)
    
    def message_percentage():
        tkMessageBox.showinfo("Percentage", perc_info)

    def print_rules():
        win = Toplevel(fenster, width=200, height=200)
        #canvas = Canvas(win, bd=0, highlightthickness=0,
                      #  yscrollcommand=vscrollbar.set, width=800, height=800)
        label1 = Label(win, text="test\ntest12\nstest\nsefawef")
        label1.pack()
        
        
    def print_rules3():
        win = Toplevel(fenster, width=2000, height=2000)
        
        mylist = Listbox(win, yscrollcommand = scrollbar.set )
        
        mylist.grid(row=0, column=0)
        
        
        vscrollbar = Scrollbar(win, orient=VERTICAL)
        canvas = Canvas(win, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set, width=800, height=800)

        vscrollbar.config(command=mylist.yview)

        text = canvas.create_text(1,2,anchor='nw', text="test")

        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0,weight=1)

        
    def print_rules2():
        global new_ruleset
        global red_ruleset
        
        win = Toplevel(fenster)
        win.title("All Rules in Data Set")
        scroll = Scrollbar(win, width=20)
        scroll.pack(side = RIGHT, fill = Y)
        scroll.grid(row=0, column=0)

        txt = Text(win, yscrollcommand=scroll.set)
        txt.grid(row=0,column=0)
        txt.insert(INSERT, build_string_ruleset(ruleset=new_ruleset, featurenames=feature_names))
        
        scroll.config(command=txt)
        

    def print_rules_():
        global new_ruleset
        global red_ruleset
        
        win = Toplevel(fenster)
        win.title("All Rules in Reduced Rule Set")
        scroll = Scrollbar(win)
        #scroll.pack(side = RIGHT, fill = Y)
        scroll.grid(row=0, column=1, sticky=N+S)
        
        txt = Text(win, wrap=WORD, yscrollcommand=scroll.set, xscrollcommand=scroll.set)
        txt.grid(row=0,column=0, sticky=N+S+E+W)
        txt.insert(INSERT, build_string_ruleset(ruleset=new_ruleset, featurenames=feature_names))

        scroll.config(command=txt.yview)    
        
        
    def bar_chart_orig_rules():
        global new_ruleset
        global red_ruleset
        
        wind = Toplevel(fenster)
        wind.title("Number of rules containing respective features in original rule set")
        
        f = Figure(figsize=(5,4), dpi=100)
        ax = f.add_subplot(111)

        data = get_number_feat_in_rules(ruleset=red_ruleset, features=range(0,22))

        ind = numpy.arange(22) 
        width = .5

        rects1 = ax.bar(ind, data, width)

        canvas = FigureCanvasTkAgg(f, master=wind)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        
    def bar_chart_red_rules():
        global new_ruleset
        global red_ruleset
        
        wind = Toplevel(fenster)
        wind.title("Number of rules containing respective features in reduced rule set")
        
        f = Figure(figsize=(5,4), dpi=100)
        ax = f.add_subplot(111)

        data = get_number_feat_in_rules(ruleset=new_ruleset, features=range(0,22))

        ind = numpy.arange(22)  # the x locations for the groups
        width = .5

        rects1 = ax.bar(ind, data, width)

        canvas = FigureCanvasTkAgg(f, master=wind)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        
    # creating main window    
    fenster = Tk()
    fenster.title("Decision Support")

    # Anweisungs-Label
    dataset = Label(fenster, text=data_set_name)
    numrules= Label(fenster, text="Number of Rules: " + str(len(ruleset)))
    feat_label = Label(fenster, text="Favourite Features (optional) ")
    perc_label = Label(fenster, text = "Percentage")
    label_f0 = Label(fenster, text = feature_names[0])
    label_f1 = Label(fenster, text=feature_names[1])
    label_f2 = Label(fenster, text = feature_names[2])
    label_f3 = Label(fenster, text=feature_names[3])
    label_f4 = Label(fenster, text = feature_names[4])
    label_f5 = Label(fenster, text= feature_names[5])
    label_f6 = Label(fenster, text = feature_names[6])
    label_f7 = Label(fenster, text= feature_names[7])
    label_f8 = Label(fenster, text = feature_names[8])
    label_f9 = Label(fenster, text=feature_names[9])
    label_f10 = Label(fenster, text = feature_names[10])
    label_f11 = Label(fenster, text=feature_names[11])
    label_f12 = Label(fenster, text = feature_names[12])
    label_f13 = Label(fenster, text=feature_names[13])
    label_f14 = Label(fenster, text = feature_names[14])
    label_f15 = Label(fenster, text= feature_names[15])
    label_f16 = Label(fenster, text = feature_names[16])
    label_f17 = Label(fenster, text= feature_names[17])
    label_f18 = Label(fenster, text = feature_names[18])
    label_f19 = Label(fenster, text=feature_names[19])
    label_f20 = Label(fenster, text = feature_names[20])
    label_f21 = Label(fenster, text=feature_names[21])
    
    red1_label = Label(fenster)
    reduce_label = Label(fenster)
    predict_label= Label(fenster)
    acc_label = Label(fenster)

    # Hier kann der Benutzer eine Eingabe machen
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

    info_feat_button = Button(fenster, text = "more info", command=message_features)
    info_perc_button = Button(fenster, text = "more info", command=message_percentage)
    info_rules_button = Button(fenster, text="Print Rules", command=print_rules_)
    
    dataset.grid(row=0, column=0, columnspan=5)
    numrules.grid(row=0, column=6, columnspan=5)

    feat_label.grid(row = 4, column = 2, columnspan=3)
    perc_label.grid(row=5, column=2, columnspan=3)
    eingabefeld.grid(row = 4, column = 4, columnspan=5)
    reduce_rule_set_button.grid(row=6, column=1, columnspan=9)
    entrytext.grid(row=5, column=4, columnspan=5)
    predict_button.grid(row = 12, column = 1, columnspan=9)
    info_rules_button.grid(row=15, column=1, columnspan=9)
    #exit_button.grid(row = 4, column = 1)
    reduce_label.grid(row = 7, column = 0, columnspan = 3)
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






####################################
####################################
## generally applicable functions ##
####################################
####################################

# extract all rules from random forest
def get_all_rules(forest) :
    li = []
    for i in forest:
        li = li + get_branch(i)
        
    return li


# returns one branch of random forest as rule
def get_branch(t):
    l=[]
    ll=[]
    tt = t.tree_
    node = 0
    depths = get_node_depths(t)
    while(node < tt.node_count):
        while(tt.feature[node] != _tree.TREE_UNDEFINED):
            l.append([tt.feature[node], "l", tt.threshold[node]])
            node=node+1
        l.append([tt.value[node][0][0],tt.value[node][0][1],0])

        ll.append(copy.deepcopy(l))
        if (node != tt.node_count-1):
            for i in range(0, abs(depths[node] - depths[node+1])):
                del l[-1]

        del l[-1] 

        if(len(l) >=1):
            l[len(l)-1][1] = 'g'

        node = node+1

    return ll

# returns depth of nodes in tree t
def get_node_depths(t):

    tt = t.tree_
    
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tt.children_left, tt.children_right, depths) 
    return np.array(depths)

import copy

# prints code for decision tree in if then else form
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print "def tree({}):".format(", ".join(feature_names))
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent, tree_.value[node])

    recurse(0, 1)


# return vector with numbers of rules with respective features in given rule set
def get_number_feat_in_rules(ruleset, features):
    vec_number_rules_containing_feature = []
    
    for i in features:
        
        vec_number_rules_containing_feature.append(rule_contains_feature(feature=i, ruleset=ruleset).count(1))
        
    return vec_number_rules_containing_feature


feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

# convert string to int list
# e.g.: "1,2,3" -> [1,2,3]
def string_to_intlist(my_string):
    str_list =  [x.strip() for x in my_string.split(',')]
    return map(int, str_list)


# calculate rule scores
# ruleset: the ruleset of which the scores should be calculated
# xtrain: the training set (dataframe)
# ytrain: the ground truth of the training set (vector)
# k: positive constant (usually 4)
# favourite_features: list of favourite features (e.g. [1,2,3])
def get_rule_score2_pers_set_newApply(ruleset, xtrain, ytrain, k, favourite_features):
    res=[]

    for x in ruleset:
        vec_cat = apply_ruleset_get_vector_new(ruleset=[x], xtest=xtrain)

        score1 = ((get_correctly_classified_2(vector=vec_cat, ytest=ytrain)-get_incorrectly_classified_2(vector=vec_cat, ytest=ytrain))/(get_correctly_classified_2(vector=vec_cat, ytest=ytrain)+get_incorrectly_classified_2(vector=vec_cat, ytest=ytrain)))+((get_correctly_classified_2(vector=vec_cat, ytest=ytrain))/(get_incorrectly_classified_2(vector=vec_cat, ytest=ytrain)+k))
        score2 = score1 + (get_correctly_classified_2(vector=vec_cat, ytest=ytrain)/len(x))
        addition = 0
        # add a new value depending on how important the feature is that is included
        # in the rule
        for i in range(0,len(favourite_features)):
            if(does_rule_contain_feature(feature=favourite_features[i], rule=x)):
                addition = addition + ((1/(i+1))*40)
                
        score_pers = score2 + addition
        res.append(score_pers)
    
    return res


#calculate rule score 1 from paper Woetzel et al.
def get_rule_score1_newApply(ruleset, xtrain, ytrain,k):
    res=[]
    
    for x in ruleset:
        # get vector with categorisations of each rule
        vec_cat = apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)
        #print(vec_cat)
        # get correct and incorrect classificatoins
        vec_compare_ytrain = compare_two_lists(list1=vec_cat, list2=ytrain)
        #print(vec_compare_ytrain)
        score = ((get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)-get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain))/(get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)+get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)))+((get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain))/(get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)+k))
        res.append(score)
    
    return res



#calculate rule score 2 from paper Woetzel et al. (considering length of rule)
def get_rule_score2_newApply(ruleset, xtrain, ytrain, k):
    res=[]

    for x in ruleset:
        # get vector with categorisations of each rule
        vec_cat = apply_rule_get_vector2(rule=x, XTest=xtrain, ytest=ytrain)

        # get correct and incorrect classificatoins
        vec_compare_ytrain = compare_two_lists(list1=vec_cat, list2=ytrain)

        score1 = ((get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)-get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain))/(get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)+get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)))+((get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain))/(get_incorrectly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)+k))
        score2 = score1 + (get_correctly_classified_2(vector=vec_compare_ytrain, ytest=ytrain)/len(x))
        res.append(score2)
    
    return 



# return new reduced rule set
# favourite_features: array with preferred features (e.g. [1,2,3])
# numtoelim: number of rules that shall be deleted
# ruleset: original rule set
# xtrain: trainings data (dataframe)
# ytrain: ground truth of training data
# k: positive constant (usually 4)
def eliminate_weakest_rules_2(favourite_features, numtoelim, ruleset, xtrain, ytrain, k):
    
    rulescores = get_rule_score2_pers_set_newApply(ruleset=ruleset, xtrain=xtrain, ytrain=ytrain, k=k, favourite_features=favourite_features)
    rules = copy.deepcopy(ruleset)
    
    for i in range(0, numtoelim):
        index = get_index_min(rulescores)
        del rules[index]
        del rulescores[index]  
    
    return rules

# returns predictions of one rule
def apply_rule_get_vector2(XTest, ytest, rule):
    #returns classification vector of one rule
    reslist = []
    
    for row in XTest.values.tolist():
        reslist.append(apply_ruleset_one_row(row, [rule]))
    
    return reslist


# get accuracy, specificity and sensitivity of rule set
# ruleset: rule set
# Xtest: trainings data (dataframe)
# ytest: ground truth of trainings data
def get_acc_spec_sens(ruleset, Xtest, ytest):
    vec = apply_ruleset_get_vector_new(ruleset=ruleset, xtest=Xtest)
    
    acc = get_accuracy_of_ruleset_new(ruleset=ruleset, xtest=Xtest, ytest=ytest)
    spec = get_specificity(reslist=vec, truevals=ytest)
    sens = get_sensitivity(reslist=vec, truevals=ytest)
    
    print("Accuracy: {}, Specificity: {}, Sensitivit: {}".format(acc, spec, sens))


# 10 fold cross validation for different values for max depth and nestimators
# df: dataframe with data
# y: vector with ground truth
# nestimators: number of trees in random forest
def ten_fold_cv(df, y, maxdepth, nestimators):
    acc_list = []
    sens_list = []
    spec_list = []
    
    for i in range(0,10):
        X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3)
        rf_test = create_random_forest(crit='gini', max_depth=maxdepth, n_estimators=nestimators, randomState=42, x_test=X_te, x_train=X_tr, y_test=y_te, y_train=y_tr)
        vec_rf_test = rf_test.predict(X_te)
        
        acc = rf_test.score(X=X_te, y=y_te)
        acc_list.append(acc)
        spec = get_specificity(reslist=vec_rf_test, truevals=y_te)
        spec_list.append(spec)
        sens = get_sensitivity(reslist=vec_rf_test, truevals=y_te)
        sens_list.append(sens)
        
        print('Number {}, acc: {}, spec: {}, sens: {}'.format(i, acc, spec, sens))
        
    print('mean acc: {}, mean sens: {}, mean spec: {}'.format(np.mean(acc_list), np.mean(sens_list), np.mean(spec_list)))



# get mean of list
def get_mean_data(liste):
    l=[]
    
    for o in liste:
        l.append(np.nanmean(o))
    
    return l


# returns accuracy of ruleset
def get_accuracy_of_ruleset_2(ruleset, xtest, ytest):
    vec = apply_ruleset_get_vector(ruleset=ruleset, xtest=xtest)
    
    numCorr = get_correctly_classified_2(vector=vec, ytest=ytest)
    
    return numCorr / (len(vec))

# returns sensitivity of rule set
def get_sensitivity_of_ruleset_2(ruleset, xtest, ytest):
    vec = apply_ruleset_get_vector(ruleset=ruleset, xtest=xtest)
    
    return get_sensitivity(reslist=vec, truevals=ytest)

# returns specificity of rule set
def get_specificity_of_ruleset_2(ruleset, xtest, ytest):
    vec = apply_ruleset_get_vector(ruleset=ruleset, xtest=xtest)
    
    return get_specificity(reslist=vec, truevals=ytest)

# returns list with all accuracies of all rules, after always a certain number
# (numtoelim) is deleted from the rule set and
# each feature is regarded as preferred one once
def get_accs_ff(ruleset, numtoelim):
    reslist = []
    for i in range(0,22):
        rules_elim = eliminate_weakest_rules_2(favourite_features=[i], numtoelim=numtoelim, ruleset=allrulesrf_red, xtrain=X_train, ytrain=y_train, k=4)
        reslist.append(get_accuracy_of_ruleset_2(ruleset=ruleset, xtest=X_test, ytest=y_test))
        
    return reslist


# get vector of number of features in certain rule sets
def distribution_featues_rules_deletion_2(ruleset, numtodel):
    vec_number_rules_containing_feature = []
    vec_number_rules_deleted_containing_feature =[]
    vec_features_deleted_normal = get_indices_deleted_rules(favourite_features=[], numtoelim=numtodel, ruleset=ruleset_split_red, xtrain=X_train, ytrain=y_train, k=4)
    vec_number_rules_deleted_containing_feature_wo_considering_favouriteF =[]
    
    for i in range(0,22):
        vec_all_rules_feature = get_indices_of_rules_with_features(ruleset=ruleset, feature=i)
        vec_number_rules_containing_feature.append(len(vec_all_rules_feature))
        
        vec_del_rules = get_indices_deleted_rules(favourite_features=[i], numtoelim=numtodel, ruleset=ruleset, xtrain=X_train, ytrain=y_train, k=4)
        vec_number_rules_deleted_containing_feature.append(len(set(vec_del_rules).intersection(vec_all_rules_feature)))
    
        vec_number_rules_deleted_containing_feature_wo_considering_favouriteF.append(len(set(vec_features_deleted_normal).intersection(vec_all_rules_feature)))
    
    return vec_number_rules_deleted_containing_feature_wo_considering_favouriteF


# returns vector of indices of rules that contain certain feature
def get_indices_of_rules_with_features(ruleset, feature):
    vec = rule_contains_feature(feature=feature, ruleset=ruleset)
    
    x = np.array(vec)
    return np.where(x == 1)[0]



def get_incorrectly_classified_2(vector, ytest):
    res =[]
    for i in range(0, len(vector)):
        if(vector[i] == ytest[i]):
            res.append(1)
        else:
            res.append(0)
        
    return res.count(0)

def get_correctly_classified_2(vector, ytest):
    res =[]
    for i in range(0, len(vector)):
        if(vector[i] == ytest[i]):
            res.append(1)
        else:
            res.append(0)
        
    return res.count(1)

# returns Ture if 'rule' contains a query based on 'feature'
def does_rule_contain_feature(feature, rule):
    for i in range(0, len(rule)-1):
        if(rule[i][0] == feature):
            return True
    return False

# convert all trees in forest to png file
def print_all_trees(forest):
    i = 0
    for x in forest:
        tree_to_png(name="tree_forest_split"+str(i), t=x)
        i= i+1


def compare_two_lists(list1, list2):
    res = []
    if(len(list1) == len(list2)):
        for i in range(0,len(list1)):
            if (list1[i] == list2[i]):
                res.append(1)
            else:
                res.append(0)
    return res



def reduce_rules(rulelist):
    todel=[]
    for i in range(0,len(rulelist)):
        #print("i: {}".format(i))
        for k in range(0, len(rulelist[i])):
            
            for j in range(1,(len(rulelist[i])-k)):
                #print("j: {}".format(j))
                if((rulelist[i][k][0] == rulelist[i][k+j][0]) & (rulelist[i][k][1]==rulelist[i][k+j][1])):

                    todel.append([i,k])
                    #print("todel: {}".format(todel))
    l=copy.deepcopy(rulelist)
    for index in sorted(todel, reverse=True):
        #print("deleting {}".format(todel))
        del l[index[0]][index[1]]
        
    return l

# applies the whole ruleset to one row (features) (ignore name column)
def apply_ruleset_one_column(features, ruleset):
    #featrues: one row in data frame X_test as list
    #ruleset: list of list of lists with all rules
    
    tmp=[]
    #i: rule number
    for i in range(0, len(ruleset)):
        tmp.append(does_rule_apply(l=features, rule=ruleset[i]))
    
    print(tmp.count(1))
    print(tmp.count(0))
    print(tmp.count(-1))
    
    if(tmp.count(1) > tmp.count(0)):
        return 1
    else:
        return 0

# check if one rule can be applied for one column of dataframe l (list)
def does_rule_apply(l, rule):
    # return 1 if diseased, 0 if not 

    # check what rule predicts, if all conditions are met
    if(rule[len(rule)-1][1] > rule[len(rule)-1][0]):
        # diseased
        tmp = 1
    else:
        # healthy
        tmp = 0

    for i in range(0,len(rule)-1):
        if(rule[i][1] == 'l'):
            if(l[int(rule[i][0])] > rule[i][2]):
                if (tmp==1):
                    return 0
                else: 
                    return 1
        else:
            if(l[int(rule[i][0])] <= rule[i][2]):
                if (tmp==1):
                    return 1
                else: 
                    return 0
    if(rule[len(rule)-1][1] > rule[len(rule)-1][0]):
        return 1
    else:
        return 0


# convert tree to png image
def tree_to_png(t, name):
    tree.export_graphviz(t, name + ".dot")
    subprocess.call(['/home/tamara/Documents/MPhilACS/Dissertation/Data/ParkinsonSpeechSignalsOxford/dot_to_png.sh', name])

def vary_depth(xtest, ytest, xtrain, ytrain):
    l=[]
    for i in range(1,20):
        rf_depth = create_random_forest_wo_print(n_estimators=30, crit="gini", max_depth=i, randomState=42, x_test=xtest, y_test=ytest)
        l.append(rf_depth.score(xtest, ytest))
    return l



def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)

# creates a random forest with given parameters
def create_random_forest(n_estimators, crit, max_depth, x_train, y_train, x_test, y_test, randomState):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=crit, random_state = randomState)
    rf.fit(x_train, y_train)

    return rf

# create test train split of specific data
def set_test_train_split() :
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(create_dataframe("dataWoName.csv"), groundTruth, test_size=0.33, random_state=42)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def bootsing_and_stuff(xtrain, ytrain, xtest, ytest):
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(random_state = 42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

    # Fit the random search model
    rf_random.fit(xtrain, ytrain);
    rf_random.best_params_
    rf_random.cv_results_

    base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
    base_model.fit(xtrain, ytrain)
    base_accuracy = evaluate(base_model, xtest, ytest)

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, xtest, ytest)

def vary_nestimators():
    i=0
    for x in participants:
        accs = testing_estimators(estimators_array=range(1,40), crit="gini",xtrain=list_X_trains[i],ytrain=list_y_trains[i], xtest=list_X_tests[i], ytest=list_y_tests[i], randomState=42)
        i=i+1

        plt.plot(accs)
        plt.savefig("plot" + str(i) + ".png")
        
        plt.clf()
        plt.cla()
        plt.close() 


# get overall accuracy and F1 score to print at top of plot
def confusion_matrix_special(y_test, pred) :
    pscore = metrics.accuracy_score(y_test, pred)
    score = metrics.f1_score(y_test, pred, pos_label=list(set(y_test)))
    # get size of the full label set
    dur = len(categories)
    print "Building testing confusion matrix..."
    # initialize score matrices
    trueScores = np.zeros(shape=(dur,dur))
    predScores = np.zeros(shape=(dur,dur))
    # populate totals
    for i in xrange(len(y_test)-1):
        trueIdx = y_test[i]
        predIdx = pred[i]
        trueScores[trueIdx,trueIdx] += 1
        predScores[trueIdx,predIdx] += 1
    # create %-based results
    trueSums = np.sum(trueScores,axis=0)
    conf = np.zeros(shape=predScores.shape)
    for i in xrange(len(predScores)):
        for j in xrange(dur):
            conf[i,j] = predScores[i,j] / trueSums[i]
    # plot the confusion matrix
    hq = pl.figure(figsize=(15,15));
    aq = hq.add_subplot(1,1,1)
    aq.set_aspect(1)
    res = aq.imshow(conf,cmap=pl.get_cmap('Greens'),interpolation='nearest',vmin=-0.05,vmax=1.)
    width = len(conf)
    height = len(conf[0])
    done = []
    # label each grid cell with the misclassification rates
    for w in xrange(width):
        for h in xrange(height):
            pval = conf[w][h]
            c = 'k'
            rais = w
            if pval > 0.5: c = 'w'
            if pval > 0.001:
                if w == h:
                    aq.annotate("{0:1.1f}%\n{1:1.0f}/{2:1.0f}".format(pval*100.,predScores[w][h],trueSums[w]), xy=(h, w), 
                      horizontalalignment='center',
                      verticalalignment='center',color=c,size=10)
            else:
                aq.annotate("{0:1.1f}%\n{1:1.0f}".format(pval*100.,predScores[w][h]), xy=(h, w), 
                      horizontalalignment='center',
                      verticalalignment='center',color=c,size=10)
    # label the axes
    pl.xticks(range(width), categories[:width],rotation=90,size=10)
    pl.yticks(range(height), categories[:height],size=10)
    # add a title with the F1 score and accuracy
    aq.set_title(lbl + " Prediction, Test Set (f1: "+"{0:1.3f}".format(score)+', accuracy: '+'{0:2.1f}%'.format(100*pscore)+", " + str(len(y_test)) + " items)",fontname='Arial',size=10,color='k')
    aq.set_ylabel("Actual",fontname='Arial',size=10,color='k')
    aq.set_xlabel("Predicted",fontname='Arial',size=10,color='k')
    pl.grid(b=True,axis='both')
    # save it
    pl.savefig("pred.conf.test.png")



# create confusion matrix with new random forest
def confusion_matrix(xtest, ytest, xtrain, ytrain):
    forest = RandomForestClassifier(n_estimators=30, criterion="gini", max_depth=100).fit(xtrain, ytrain)
    predicted1 = forest.predict(xtest)
    accuracy = accuracy_score(ytest, predicted1)
    cm = pd.DataFrame(confusion_matrix(ytest, predicted1,labels=[0, 1]))
    sns.heatmap(cm, annot=True);


def sum_up_accuracy(ll, varyDepth):
    #returns list of sums of accuracies; immer erste accuracy (also mit depth 1) von allen forests aufaddieren, und dann mit depth2, also ll[1] and so on
    list_of_sums = []
    for i in range(0, varyDepth-1):
        summe = 0
        print("Depth: {}".format(i))
        #ll[i] ist Liste der Accuracies des ersten Forests (without Participant 1)
        for j in range(0,(len(ll)-1)):
            summe = summe + ll[j][i]
        list_of_sums.append(summe)
    return list_of_sums



################################################
################################################
############# tree-like approach ###############
### like decision trees, only one prediction ###
################################################
################################################



# returns bool value
# True if the rule applies for row
# False if not
def does_rule_apply2(rule, row):
    
    for i in range(0,len(rule)-1):
        if(rule[i][1] == 'l'):
            if(row[int(rule[i][0])] > rule[i][2]):
                return False
        else:
            if(row[int(rule[i][0])] <= rule[i][2]):
                return False
            
    return True

# get prediciton of ruleset for one row
# ruleset: array of array of array with rules
# row: one row in dataframe
def apply_ruleset_one_row_new(ruleset, row):
    res =[]
    for rule in ruleset:
        if(does_rule_apply2(row=row, rule=rule)):
            if(rule[len(rule)-1][0] > rule[len(rule)-1][1]):
                res.append(0)
            else:
                res.append(1)
               
    if(res.count(0) > res.count(1)):
        return 0
    else:
        return 1

# returns accuracy of a rule set
def get_accuracy_of_ruleset_new(ruleset, xtest, ytest):
    vec = apply_ruleset_get_vector_new(ruleset=ruleset, xtest=xtest)
    
    if(len(vec) != len(ytest)):
        print("not the same length")
        return (-1)
    
    correct = get_correctly_classified_2(vector=vec, ytest=ytest)
    
    return correct/len(vec)


# returns vector of predictions for xteset based on the rule set
def apply_ruleset_get_vector_new(ruleset, xtest):
    res =[]
    for row in xtest.values.tolist():
        res.append(apply_ruleset_one_row_new(ruleset=ruleset, row=row))
    return res


def compare_two_lists(list1, list2):
    res = []
    if(len(list1) == len(list2)):
        for i in range(0,len(list1)):
            if (list1[i] == list2[i]):
                res.append(1)
            else:
                res.append(0)
    return res







#################################
#################################
###### TREE BASED APPROACH ######
## each tree stored separately ##
#################################
#################################


# extract all rules from random forest
# separate by decision tree
# that way only one result is calculated for each tree,
# that should result in better results
def get_all_rules2(forest) :
    li = []
    for i in forest:
        li.append(get_branch(i))
        
    return li


# checks whether a rule is applicable 
# only returns true or false
def does_rule_apply2(rule, row):
    
    for i in range(0,len(rule)-1):
        if(rule[i][1] == 'l'):
            if(row[int(rule[i][0])] > rule[i][2]):
                return False
        else:
            if(row[int(rule[i][0])] <= rule[i][2]):
                return False
            
    return True

# This method applies a rule set ot one row with the tree based ruleset
# always checking whether a rule is applicable and only evaluating the one rule within a tree
# that is applicable to the data sample
def apply_ruleset_one_row_treeBased(ruleset, row):
    res = []
    for tree in ruleset:
        
        for rule in tree:
            if(does_rule_apply2(row=row, rule=rule)):
                if(rule[len(rule)-1][0] > rule[len(rule)-1][1]):
                    res.append(0)
                else:
                    res.append(1)

    if(res.count(0) > res.count(1)):
        return 0
    else:
        return 1

# returns the vecotr of results with all values for all samples in xteset
# this is tree based, so the ruleset stores each decision tree separately
def apply_ruleset_get_vector_treeBased(ruleset, xtest):
    res = []
    for row in xtest.values.tolist():
        res.append(apply_ruleset_one_row_treeBased(ruleset=ruleset, row=row))
    return res

#returns accuracy of ruleset based on xtest and ytest
# (number of correctly classified devidied by number of all samples)
def get_accuracy_of_ruleset_treeBased(ruleset, Xtest, ytest):
    vec = apply_ruleset_get_vector_treeBased(ruleset=ruleset, xtest=Xtest)
    
    if(len(vec) != len(ytest)):
        print("not the same length")
        return -1
    
    correct = get_correctly_classified_2(vector=vec, ytest=ytest)
    
    return correct/len(vec)

# this method eliminates useless queries within one branch of a tree
# if one rules performs two checks on the same feature with the same condition
# one of these queries can be removed:
# if it checks whether feature 1 is smaller than 10 and later it checks whether 
# feature 1 is smaller than 5, the smaller than 10 querie can be removed without
# changing the outcome of the rule
def reduce_rules_treeBased(ruleset):
    todel=[]
    #trees
    a = 0
    for tree in ruleset:
        #rules in tree
        for i in range(0,len(tree)):
            #print("i: {}".format(i))
            for k in range(0, len(tree[i])):

                for j in range(1,(len(tree[i])-k)):
                    #print("j: {}".format(j))
                    if((tree[i][k][0] == tree[i][k+j][0]) & (tree[i][k][1]==tree[i][k+j][1])):

                        todel.append([a,i,k])
        a=a+1

    l=copy.deepcopy(ruleset)
    for index in sorted(todel, reverse=True):
        del l[index[0]][index[1]][index[2]]
        
    return l

# tree based ruleset -> sum up number of all rules
def get_number_of_rules(ruleset):
    res = 0
    
    for tree in ruleset:
        res = res+ len(tree)
        
    return res

# get ranking of all trees
def rank_decision_trees_on_accuracy(ruleset, xtest, ytest):
    res = []
    for tree in ruleset:
        res.append(get_accuracy_of_ruleset_treeBased(ruleset=[tree], Xtest=xtest, ytest=ytest))
        
    return res
    

def get_specificity_treeBased(ruleset, Xtest, ytest):
    vec = apply_ruleset_get_vector_treeBased(ruleset=ruleset, xtest=Xtest)
    
    if(len(vec) != len(ytest)):
        print("not the same length")
        return -1

    return get_specificity(reslist=vec, truevals=ytest)

def get_sensitivity_treeBased(ruleset, Xtest, ytest):
    vec = apply_ruleset_get_vector_treeBased(ruleset=ruleset, xtest=Xtest)
    
    if(len(vec) != len(ytest)):
        print("not the same length")
        return -1

    return get_sensitivity(reslist=vec, truevals=ytest)


# delete rules that are almost the same, except last query
# only possible with tree-like application
def delete_useless_rules(ruleset):
    rules = copy.deepcopy(ruleset)
    vectodel = []
    boolval = True
    for i in range(0, len(ruleset)-1):
        k=0
        if(len(ruleset[i]) == len(ruleset[i+1])): 
            j=0
            while (k != -1):
                if(j==len(ruleset[i])-2):
                    if((ruleset[i][j][0] == ruleset[i+1][j][0]) & 
                       (ruleset[i][j][2] == ruleset[i+1][j][2]) &
                       (ruleset[i][j][1] != ruleset[i+1][j][1]) ):
                        #print(i+1)
                        vectodel.append(i+1)
                        j=j+1
                        k=-1
                elif(ruleset[i][j] == ruleset[i+1][j]):
                    j=j+1
                else: 
                    k=-1
                    
    
    #print(vectodel)
    
    for index in sorted(vectodel, reverse=True):
        del rules[index]
    
    return rules


############################
############################
## working with MR images ##
####### in ADNI data #######
############################
############################


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


AD_002_S_0619__1 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR__GradWarp__N3__Scaled_Br_20070717184209073_S24022_I60451.nii')
AD_002_S_0619__2 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_2_Br_20081001115218896_S15145_I118678.nii')
AD_002_S_0619__3 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_Br_20070411125458928_S15145_I48617.nii')
AD_002_S_0619__4 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_Br_20070816100717385_S33969_I67871.nii')
AD_002_S_0816__1 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081002102135862_S18402_I118984.nii')
AD_002_S_0816__2 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217005829488_S18402_I40731.nii')
AD_002_S_0816__3 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070717185335251_S29612_I60465.nii')
AD_002_S_0816__4 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080224131120787_S45030_I92146.nii')
AD_002_S_0938__1 = load_image(filename='0_AD/ADNI_002_S_0938_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219175406282_S19852_I40980.nii')
AD_002_S_1018__1 = load_image(filename='0_AD/ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217030439623_S23128_I40817.nii')
AD_005_S_0221__1 = load_image(filename='0_AD/ADNI_005_S_0221_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080410142121502_S28459_I102054.nii')
AD_005_S_0221__2 = load_image(filename='0_AD/ADNI_005_S_0221_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20061212173139396_S19846_I32899.nii')
AD_007_S_0316__1 = load_image(filename='0_AD/ADNI_007_S_0316_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070118024327307_S21488_I36559.nii')
AD_007_S_0316__2 = load_image(filename='0_AD/ADNI_007_S_0316_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070923130250878_S31849_I74627.nii')
AD_007_S_1339__1 = load_image(filename='0_AD/ADNI_007_S_1339_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070607134807952_S27414_I56319.nii')
AD_007_S_1339__2 = load_image(filename='0_AD/ADNI_007_S_1339_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071027133744003_S41402_I78754.nii')
AD_007_S_1339__3 = load_image(filename='0_AD/ADNI_007_S_1339_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080430143918869_S49074_I104363.nii')
AD_010_S_0786__1 = load_image(filename='0_AD/ADNI_010_S_0786_MR_MPR____N3__Scaled_2_Br_20081002102855696_S19638_I118990.nii')
AD_010_S_0829__1 = load_image(filename='0_AD/ADNI_010_S_0829_MR_MPR____N3__Scaled_Br_20080410112243910_S46875_I101977.nii')
AD_011_S_0003__1 = load_image(filename='0_AD/ADNI_011_S_0003_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070109190839611_S19096_I35576.nii')
AD_011_S_0010__1 = load_image(filename='0_AD/ADNI_011_S_0010_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080229151344346_S45936_I94368.nii')
AD_011_S_0053__1 = load_image(filename='0_AD/ADNI_011_S_0053_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070810172103015_S23166_I66945.nii')
AD_067_S_0076__1 = load_image(filename='0_AD/ADNI_067_S_0076_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081003155853706_S16974_I119191.nii')
AD_067_S_0076__2 = load_image(filename='0_AD/ADNI_067_S_0076_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061229172001056_S16974_I34788.nii')
AD_094_S_1164__1 = load_image(filename='0_AD/ADNI_094_S_1164_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080220140841993_S44407_I91057.nii')
AD_094_S_1164__2 = load_image(filename='0_AD/ADNI_094_S_1164_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081016191406569_S51749_I121632.nii')
AD_094_S_1397__1 = load_image(filename='0_AD/ADNI_094_S_1397_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20120731130350494_S50305_I319849.nii')
AD_094_S_1397__2 = load_image(filename='0_AD/ADNI_094_S_1397_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080307104443256_S31011_I95662.nii')


NC_002_S_0295__1 = load_image(filename='0_NC/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120532722_S21856_I118692.nii')
NC_002_S_0295__2 = load_image(filename='0_NC/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii')
NC_002_S_0295__3 = load_image(filename='0_NC/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319113623975_S13408_I45108.nii')
NC_002_S_0413__1 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114742166_S13893_I118673.nii')
NC_002_S_0413__2 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120813046_S22557_I118695.nii')
NC_002_S_0413__3 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319115331858_S13893_I45117.nii')
NC_002_S_0413__4 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070713121420365_S32938_I60008.nii')
NC_002_S_0413__5 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071028190559976_S22557_I79122.nii')
NC_002_S_0685__1 = load_image(filename='0_NC/ADNI_002_S_0685_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20071223122419058_S25369_I86020.nii')
NC_002_S_1261__1 = load_image(filename='0_NC/ADNI_002_S_1261_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080613103237200_S50898_I109394.nii')
NC_002_S_1280__1 = load_image(filename='0_NC/ADNI_002_S_1280_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071110105845543_S38235_I81321.nii')
NC_003_S_0981__1 = load_image(filename='0_NC/ADNI_003_S_0981_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080131105013455_S31564_I89046.nii')
NC_005_S_0223__1 = load_image(filename='0_NC/ADNI_005_S_0223_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080410141453831_S28246_I102045.nii')
NC_006_S_0731__1 = load_image(filename='0_NC/ADNI_006_S_0731_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070804130615519_S27980_I64575.nii')
NC_007_S_0068__1 = load_image(filename='0_NC/ADNI_007_S_0068_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070110214628818_S23109_I35773.nii')
NC_007_S_0070__1 = load_image(filename='0_NC/ADNI_007_S_0070_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070118014059657_S18818_I36523.nii')
NC_007_S_1206__1 = load_image(filename='0_NC/ADNI_007_S_1206_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070923131606579_S38141_I74645.nii')
NC_010_S_0067__1 = load_image(filename='0_NC/ADNI_010_S_0067_MR_MPR____N3__Scaled_2_Br_20081001122414391_S25341_I118706.nii')
NC_010_S_0419__1 = load_image(filename='0_NC/ADNI_010_S_0419_MR_MPR____N3__Scaled_Br_20070731161135546_S24112_I63325.nii')
NC_011_S_0005__1 = load_image(filename='0_NC/ADNI_011_S_0005_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061206162204955_S12037_I31885.nii')
NC_011_S_0021__1 = load_image(filename='0_NC/ADNI_011_S_0021_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061206174058589_S22065_I31970.nii')
NC_011_S_0023__1 = load_image(filename='0_NC/ADNI_011_S_0023_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070808160104173_S23153_I65902.nii')
NC_067_S_0056__1 = load_image(filename='0_NC/ADNI_067_S_0056_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081003155535571_S8723_I119186.nii')
NC_067_S_0056__2 = load_image(filename='0_NC/ADNI_067_S_0056_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070209152131877_S16358_I38678.nii')
NC_094_S_1241__1 = load_image(filename='0_NC/ADNI_094_S_1241_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081016163336876_S50440_I121470.nii')
NC_094_S_1241__2 = load_image(filename='0_NC/ADNI_094_S_1241_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20081016181908278_S38195_I121579.nii')

# create lists of data and names
list_names_AD = ['AD_002_S_0619__1', 'AD_002_S_0619__2', 'AD_002_S_0619__3', 'AD_002_S_0619__4', 'AD_002_S_0816__1', 'AD_002_S_0816__2', 'AD_002_S_0816__3', 'AD_002_S_0816__4', 'AD_002_S_0938__1', 'AD_002_S_1018__1', 'AD_005_S_0221__1', 'AD_005_S_0221__2', 'AD_007_S_0316__1', 'AD_007_S_0316__2', 'AD_007_S_1339__1', 'AD_007_S_1339__2', 'AD_007_S_1339__3', 'AD_010_S_0786__1', 'AD_010_S_0829__1', 'AD_011_S_0003__1', 'AD_011_S_0010__1', 'AD_011_S_0053__1', 'AD_067_S_0076__1', 'AD_067_S_0076__2', 'AD_094_S_1164__1', 'AD_094_S_1164__2', 'AD_094_S_1397__1', 'AD_094_S_1397__2']
list_AD = [AD_002_S_0619__1, AD_002_S_0619__2, AD_002_S_0619__3, AD_002_S_0619__4, AD_002_S_0816__1, AD_002_S_0816__2, AD_002_S_0816__3, AD_002_S_0816__4, AD_002_S_0938__1, AD_002_S_1018__1, AD_005_S_0221__1, AD_005_S_0221__2, AD_007_S_0316__1, AD_007_S_0316__2, AD_007_S_1339__1, AD_007_S_1339__2, AD_007_S_1339__3, AD_010_S_0786__1, AD_010_S_0829__1, AD_011_S_0003__1, AD_011_S_0010__1, AD_011_S_0053__1, AD_067_S_0076__1, AD_067_S_0076__2, AD_094_S_1164__1, AD_094_S_1164__2, AD_094_S_1397__1, AD_094_S_1397__2]
list_names_NC = ['NC_002_S_0295__1', 'NC_002_S_0295__2', 'NC_002_S_0295__3', 'NC_002_S_0413__1', 'NC_002_S_0413__2', 'NC_002_S_0413__3', 'NC_002_S_0413__4', 'NC_002_S_0413__5', 'NC_002_S_0685__1', 'NC_002_S_1261__1', 'NC_002_S_1280__1', 'NC_003_S_0981__1', 'NC_005_S_0223__1', 'NC_006_S_0731__1', 'NC_007_S_0068__1', 'NC_007_S_0070__1', 'NC_007_S_1206__1', 'NC_010_S_0067__1', 'NC_010_S_0419__1', 'NC_011_S_0005__1', 'NC_011_S_0021__1', 'NC_011_S_0023__1', 'NC_067_S_0056__1', 'NC_067_S_0056__2', 'NC_094_S_1241__1', 'NC_094_S_1241__2']
list_NC= [NC_002_S_0295__1, NC_002_S_0295__2, NC_002_S_0295__3, NC_002_S_0413__1, NC_002_S_0413__2, NC_002_S_0413__3, NC_002_S_0413__4, NC_002_S_0413__5, NC_002_S_0685__1, NC_002_S_1261__1, NC_002_S_1280__1, NC_003_S_0981__1, NC_005_S_0223__1, NC_006_S_0731__1, NC_007_S_0068__1, NC_007_S_0070__1, NC_007_S_1206__1, NC_010_S_0067__1, NC_010_S_0419__1, NC_011_S_0005__1, NC_011_S_0021__1, NC_011_S_0023__1, NC_067_S_0056__1, NC_067_S_0056__2, NC_094_S_1241__1, NC_094_S_1241__2]

# get array of MR image
def get_data(names):
    l = []
    for n in names:
        l.append(n.get_data())
    
    return l

def get_sumsumsumdata(obj):
    l=[]
    for o in obj:
        l.append(sum(sum(sum(o))))
    
    return l

def get_mean_data(liste):
    l=[]
    
    for o in liste:
        l.append(np.nanmean(o))
    
    return l

def get_maxsumsum_data(liste):
    l=[]
    for o in liste:
        l.append(max(sum(sum(o))))
    return l

def get_slice0(liste):
    l = []
    
    for o in liste:
        fst = o.shape[0]

        l.append(o[(fst//2), :, :])
    
    return l

def get_slice1(liste):
    l=[]
    for o in liste:
        snd = o.shape[1]
        l.append(o[:, snd//2, :])
    return l

def get_slice2(liste):
    l=[]
    for o in liste:
        trd = o.shape[2]
        l.append(o[:, :, trd//2])
    return l

def get_sumsumsslice(liste):
    l=[]
    for o in liste:
        l.append(sum(sum(o)))
    return l

def get_meanslice(liste):
    l=[]
    for o in liste:
        l.append(np.nanmean(o))
    return l

def get_maxsumslice(liste):
    l =[]
    for o in liste:
        l.append(max(sum(o)))
                 
    return l

def get_meansumslice(liste):
    l=[]
    for o in liste:
        l.append(np.nanmean(sum(o)))
    return l

def get_maxsum(liste):
    l = []
    
    for o in liste:
        l.append(max(o))
        
    return l

def get_flat_list(liste):
    flat_list =[]
    for sublist in liste:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def get_maxflat(liste):
    
    l = []
    for o in liste:
        list_flat = get_flat_list(o)
        l.append(max(list_flat))
        
    return l





# get probabilites of tissues in brain for CN (normal control group)
def get_tissue_sums_CN_smooth(list_data):
    global white_CN_smooth
    global gray_CN_smooth
    global csf_CN_smooth
    global white_sum_CN_smooth
    global gray_sum_CN_smooth
    global csf_sum_CN_smooth
    
    i=0
    for o in list_data:
        try:
            hmrf = TissueClassifierHMRF()
            initial_segmentation, final_segmentation, PVE = hmrf.classify(o, nclass, beta)

            img_ax = np.rot90(final_segmentation[..., 89])
            img_cor = np.rot90(final_segmentation[:, 128, :])

            img_ax = np.rot90(PVE[..., 89, 0])
            #CSF
            csf_CN_smooth.append(img_ax)
            csf_sum_CN_smooth.append(sum(img_ax))

            img_cor = np.rot90(PVE[:, :, 89, 1])
            #GRAY
            gray_CN_smooth.append(img_cor)
            gray_sum_CN_smooth.append(sum(img_cor))

            img_cor = np.rot90(PVE[:, :, 89, 2])
            #WHITE
            white_CN_smooth.append(img_cor)
            white_sum_CN_smooth.append(sum(img_cor))
            i=i+1
        except ValueError as error:
            i=i+1
            print error
            print i



# exemplary usage of functions above to extract features from MR images
def create_features():
	list_data_AD = get_data(list_AD)
	list_data_NC = get_data(list_NC)
	list_sumsumsumAD = get_sumsumsumdata(list_data_AD)


	x = sum(sum(AD_002_S_0619__3.get_data()))
	y = sum(sum(AD_002_S_0816__2.get_data()))
	x[np.isnan(x)]=0
	y[np.isnan(y)]=0
	y[np.isinf(y)]=0
	list_sumsumsumAD[2] = sum(x)
	list_sumsumsumAD[5] = sum(y)

	list_sumsumsumNC = get_sumsumsumdata(list_data_NC)
	z = sum(sum(NC_002_S_0295__2.get_data()))
	z[np.isnan(z)]=0
	list_sumsumsumNC[1] = sum(z)

	list_mean_AD = get_mean_data(list_data_AD)

	list_mean_AD[5] = np.nanmean(list_mean_AD)

	list_mean_NC = get_mean_data(list_data_NC)
	list_mean_NC[1] = np.nanmean(list_mean_NC)

	list_maxsumsum_AD = get_maxsumsum_data(list_data_AD)
	del list_maxsumsum_AD[5]
	tmp = np.mean(list_maxsumsum_AD)
	list_maxsumsum_AD = get_maxsumsum_data(list_data_AD)
	list_maxsumsum_AD[5] = tmp

	list_maxsumsum_NC = get_maxsumsum_data(list_data_NC)

	list_slice0_AD = get_slice0(list_data_AD)
	list_slice0_NC = get_slice0(list_data_NC)

	list_slice1_AD = get_slice1(list_data_AD)
	list_slice1_NC = get_slice1(list_data_NC)

	list_slice2_AD = get_slice2(list_data_AD)
	list_slice2_NC = get_slice2(list_data_NC)

	list_sumslice0_AD = get_sumsumsslice(list_slice0_AD)
	list_sumslice0_NC = get_sumsumsslice(list_slice0_NC)
	list_sumslice0_NC[1] = np.nanmean(list_sumslice0_NC)

	list_sumslice1_AD = get_sumsumsslice(list_slice1_AD)
	list_sumslice1_NC = get_sumsumsslice(list_slice1_NC)
	list_sumslice1_NC[1] = np.nanmean(list_sumslice0_NC)

	list_sumslice2_AD = get_sumsumsslice(list_slice2_AD)
	list_sumslice2_NC = get_sumsumsslice(list_slice2_NC)

	list_meanslice0_AD = get_meanslice(list_slice0_AD)
	list_meanslice0_NC = get_meanslice(list_slice0_NC)
	list_meanslice1_AD = get_meanslice(list_slice1_AD)
	list_meanslice1_NC = get_meanslice(list_slice1_NC)
	list_meanslice2_AD = get_meanslice(list_slice2_AD)
	list_meanslice2_NC = get_meanslice(list_slice2_NC)

	list_maxsumslice0_AD = get_maxsumslice(list_slice0_AD)
	list_maxsumslice0_NC = get_maxsumslice(list_slice0_NC)
	del list_maxsumslice0_NC[1]
	tmp2 = np.mean(list_maxsumslice0_NC)
	list_maxsumslice0_NC[1] = tmp2

	list_maxsumslice1_AD = get_maxsumslice(list_slice1_AD)
	list_maxsumslice1_NC = get_maxsumslice(list_slice1_NC)
	list_maxsumslice2_AD = get_maxsumslice(list_slice2_AD)
	list_maxsumslice2_NC = get_maxsumslice(list_slice2_NC)


	list_meansumslice0_AD = get_meansumslice(list_slice0_AD)
	list_meansumslice0_NC = get_meansumslice(list_slice0_NC)
	del list_meansumslice0_NC[1]
	tmp3 = np.nanmean(list_meansumslice0_NC)
	list_meansumslice0_NC[1] = tmp3
	list_meansumslice1_AD = get_meansumslice(list_slice1_AD)
	list_meansumslice1_NC = get_meansumslice(list_slice1_NC)
	del list_meansumslice1_NC[1]
	tmp4 = np.mean(list_meansumslice1_NC)
	list_meansumslice1_NC[1] = tmp4
	list_mean_sumslice2_AD = get_meansumslice(list_slice2_AD)
	list_mean_sumslice2_NC = get_meansumslice(list_slice2_NC)

def get_lists_of_MR_data():
	list_names_AD_smoothed2 = ['AD_005_S_0221__3_smoothed', 'AD_005_S_0814__1_smoothed', 'AD_005_S_1341__1_smoothed', 'AD_005_S_1341__2_smoothed', 'AD_067_S_0076__3_smoothed', 'AD_068_S_0109__1_smoothed', 'AD_068_S_0109__2_smoothed', 'AD_082_S_1377__1_smoothed', 'AD_082_S_1377__2_smoothed', 'AD_094_S_1090__1_smoothed', 'AD_094_S_1090__2_smoothed', 'AD_094_S_1397__3_smoothed']
	list_names_NC_smoothed2 = ['NC_002_S_0685__2_smoothed', 'NC_002_S_0685__3_smoothed', 'NC_002_S_1261__2_smoothed', 'NC_002_S_1280__2_smoothed', 'NC_003_S_0907__1_smoothed', 'NC_005_S_0223__2_smoothed', 'NC_005_S_0553__1_smoothed', 'NC_068_S_0127__1_smoothed', 'NC_068_S_0127__2_smoothed', 'NC_068_S_0210__1_smoothed', 'NC_094_S_0692__1_smoothed']
	list_AD_smoothed2 = [AD_005_S_0221__3_smoothed, AD_005_S_0814__1_smoothed, AD_005_S_1341__1_smoothed, AD_005_S_1341__2_smoothed, AD_067_S_0076__3_smoothed, AD_068_S_0109__1_smoothed, AD_068_S_0109__2_smoothed, AD_082_S_1377__1_smoothed, AD_082_S_1377__2_smoothed, AD_094_S_1090__1_smoothed, AD_094_S_1090__2_smoothed, AD_094_S_1397__3_smoothed]
	list_NC_smoothed2 = [NC_002_S_0685__2_smoothed, NC_002_S_0685__3_smoothed, NC_002_S_1261__2_smoothed, NC_002_S_1280__2_smoothed, NC_003_S_0907__1_smoothed, NC_005_S_0223__2_smoothed, NC_005_S_0553__1_smoothed, NC_068_S_0127__1_smoothed, NC_068_S_0127__2_smoothed, NC_068_S_0210__1_smoothed, NC_094_S_0692__1_smoothed]
	list_AD2 = [AD_005_S_0221__3, AD_005_S_0814__1, AD_005_S_1341__1, AD_005_S_1341__2, AD_067_S_0076__3, AD_068_S_0109__1, AD_068_S_0109__2, AD_082_S_1377__1, AD_082_S_1377__2, AD_094_S_1090__1, AD_094_S_1090__2, AD_094_S_1397__3 ]
	list_NC2 = [NC_002_S_0685__2, NC_002_S_0685__3, NC_002_S_1261__2, NC_002_S_1280__2, NC_003_S_0907__1, NC_005_S_0223__2, NC_005_S_0553__1, NC_068_S_0127__1, NC_068_S_0127__2, NC_068_S_0210__1, NC_094_S_0692__1]
	list_names_AD2 = ['AD_005_S_0221__3', 'AD_005_S_0814__1', 'AD_005_S_1341__1', 'AD_005_S_1341__2', 'AD_067_S_0076__3', 'AD_068_S_0109__1', 'AD_068_S_0109__2', 'AD_082_S_1377__1', 'AD_082_S_1377__2', 'AD_094_S_1090__1', 'AD_094_S_1090__2', 'AD_094_S_1397__3']
	list_names_NC2 = ['NC_002_S_0685__2', 'NC_002_S_0685__3', 'NC_002_S_1261__2', 'NC_002_S_1280__2', 'NC_003_S_0907__1', 'NC_005_S_0223__2', 'NC_005_S_0553__1', 'NC_068_S_0127__1', 'NC_068_S_0127__2', 'NC_068_S_0210__1', 'NC_094_S_0692__1']
	list_names_AD_smoothed = ['AD_002_S_0619__1_smoothed', 'AD_002_S_0619__2_smoothed', 'AD_002_S_0619__3_smoothed', 'AD_002_S_0619__4_smoothed', 'AD_002_S_0816__1_smoothed', 'AD_002_S_0816__2_smoothed', 'AD_002_S_0816__3_smoothed', 'AD_002_S_0816__4_smoothed', 'AD_002_S_0938__1_smoothed', 'AD_002_S_1018__1_smoothed', 'AD_005_S_0221__1_smoothed', 'AD_005_S_0221__2_smoothed', 'AD_007_S_0316__1_smoothed', 'AD_007_S_0316__2_smoothed', 'AD_007_S_1339__1_smoothed', 'AD_007_S_1339__2_smoothed', 'AD_007_S_1339__3_smoothed', 'AD_010_S_0786__1_smoothed', 'AD_010_S_0829__1_smoothed', 'AD_011_S_0003__1_smoothed', 'AD_011_S_0010__1_smoothed', 'AD_011_S_0053__1_smoothed', 'AD_067_S_0076__1_smoothed', 'AD_067_S_0076__2_smoothed', 'AD_094_S_1164__1_smoothed', 'AD_094_S_1164__2_smoothed', 'AD_094_S_1397__1_smoothed', 'AD_094_S_1397__2_smoothed']
	list_AD_smoothed = [AD_002_S_0619__1_smoothed, AD_002_S_0619__2_smoothed, AD_002_S_0619__3_smoothed, AD_002_S_0619__4_smoothed, AD_002_S_0816__1_smoothed, AD_002_S_0816__2_smoothed, AD_002_S_0816__3_smoothed, AD_002_S_0816__4_smoothed, AD_002_S_0938__1_smoothed, AD_002_S_1018__1_smoothed, AD_005_S_0221__1_smoothed, AD_005_S_0221__2_smoothed, AD_007_S_0316__1_smoothed, AD_007_S_0316__2_smoothed, AD_007_S_1339__1_smoothed, AD_007_S_1339__2_smoothed, AD_007_S_1339__3_smoothed, AD_010_S_0786__1_smoothed, AD_010_S_0829__1_smoothed, AD_011_S_0003__1_smoothed, AD_011_S_0010__1_smoothed, AD_011_S_0053__1_smoothed, AD_067_S_0076__1_smoothed, AD_067_S_0076__2_smoothed, AD_094_S_1164__1_smoothed, AD_094_S_1164__2_smoothed, AD_094_S_1397__1_smoothed, AD_094_S_1397__2_smoothed]
	list_names_NC_smoothed = ['NC_002_S_0295__1_smoothed', 'NC_002_S_0295__2_smoothed', 'NC_002_S_0295__3_smoothed', 'NC_002_S_0413__1_smoothed', 'NC_002_S_0413__2_smoothed', 'NC_002_S_0413__3_smoothed', 'NC_002_S_0413__4_smoothed', 'NC_002_S_0413__5_smoothed', 'NC_002_S_0685__1_smoothed', 'NC_002_S_1261__1_smoothed', 'NC_002_S_1280__1_smoothed', 'NC_003_S_0981__1_smoothed', 'NC_005_S_0223__1_smoothed', 'NC_006_S_0731__1_smoothed', 'NC_007_S_0068__1_smoothed', 'NC_007_S_0070__1_smoothed', 'NC_007_S_1206__1_smoothed', 'NC_010_S_0067__1_smoothed', 'NC_010_S_0419__1_smoothed', 'NC_011_S_0005__1_smoothed', 'NC_011_S_0021__1_smoothed', 'NC_011_S_0023__1_smoothed', 'NC_067_S_0056__1_smoothed', 'NC_067_S_0056__2_smoothed', 'NC_094_S_1241__1_smoothed', 'NC_094_S_1241__2_smoothed']
	list_NC_smoothed = [NC_002_S_0295__1_smoothed, NC_002_S_0295__2_smoothed, NC_002_S_0295__3_smoothed, NC_002_S_0413__1_smoothed, NC_002_S_0413__2_smoothed, NC_002_S_0413__3_smoothed, NC_002_S_0413__4_smoothed, NC_002_S_0413__5_smoothed, NC_002_S_0685__1_smoothed, NC_002_S_1261__1_smoothed, NC_002_S_1280__1_smoothed, NC_003_S_0981__1_smoothed, NC_005_S_0223__1_smoothed, NC_006_S_0731__1_smoothed, NC_007_S_0068__1_smoothed, NC_007_S_0070__1_smoothed, NC_007_S_1206__1_smoothed, NC_010_S_0067__1_smoothed, NC_010_S_0419__1_smoothed, NC_011_S_0005__1_smoothed, NC_011_S_0021__1_smoothed, NC_011_S_0023__1_smoothed, NC_067_S_0056__1_smoothed, NC_067_S_0056__2_smoothed, NC_094_S_1241__1_smoothed, NC_094_S_1241__2_smoothed]
	list_names_AD = ['AD_002_S_0619__1', 'AD_002_S_0619__2', 'AD_002_S_0619__3', 'AD_002_S_0619__4', 'AD_002_S_0816__1', 'AD_002_S_0816__2', 'AD_002_S_0816__3', 'AD_002_S_0816__4', 'AD_002_S_0938__1', 'AD_002_S_1018__1', 'AD_005_S_0221__1', 'AD_005_S_0221__2', 'AD_007_S_0316__1', 'AD_007_S_0316__2', 'AD_007_S_1339__1', 'AD_007_S_1339__2', 'AD_007_S_1339__3', 'AD_010_S_0786__1', 'AD_010_S_0829__1', 'AD_011_S_0003__1', 'AD_011_S_0010__1', 'AD_011_S_0053__1', 'AD_067_S_0076__1', 'AD_067_S_0076__2', 'AD_094_S_1164__1', 'AD_094_S_1164__2', 'AD_094_S_1397__1', 'AD_094_S_1397__2']
	list_AD = [AD_002_S_0619__1, AD_002_S_0619__2, AD_002_S_0619__3, AD_002_S_0619__4, AD_002_S_0816__1, AD_002_S_0816__2, AD_002_S_0816__3, AD_002_S_0816__4, AD_002_S_0938__1, AD_002_S_1018__1, AD_005_S_0221__1, AD_005_S_0221__2, AD_007_S_0316__1, AD_007_S_0316__2, AD_007_S_1339__1, AD_007_S_1339__2, AD_007_S_1339__3, AD_010_S_0786__1, AD_010_S_0829__1, AD_011_S_0003__1, AD_011_S_0010__1, AD_011_S_0053__1, AD_067_S_0076__1, AD_067_S_0076__2, AD_094_S_1164__1, AD_094_S_1164__2, AD_094_S_1397__1, AD_094_S_1397__2]
	list_names_NC = ['NC_002_S_0295__1', 'NC_002_S_0295__2', 'NC_002_S_0295__3', 'NC_002_S_0413__1', 'NC_002_S_0413__2', 'NC_002_S_0413__3', 'NC_002_S_0413__4', 'NC_002_S_0413__5', 'NC_002_S_0685__1', 'NC_002_S_1261__1', 'NC_002_S_1280__1', 'NC_003_S_0981__1', 'NC_005_S_0223__1', 'NC_006_S_0731__1', 'NC_007_S_0068__1', 'NC_007_S_0070__1', 'NC_007_S_1206__1', 'NC_010_S_0067__1', 'NC_010_S_0419__1', 'NC_011_S_0005__1', 'NC_011_S_0021__1', 'NC_011_S_0023__1', 'NC_067_S_0056__1', 'NC_067_S_0056__2', 'NC_094_S_1241__1', 'NC_094_S_1241__2']
	list_NC= [NC_002_S_0295__1, NC_002_S_0295__2, NC_002_S_0295__3, NC_002_S_0413__1, NC_002_S_0413__2, NC_002_S_0413__3, NC_002_S_0413__4, NC_002_S_0413__5, NC_002_S_0685__1, NC_002_S_1261__1, NC_002_S_1280__1, NC_003_S_0981__1, NC_005_S_0223__1, NC_006_S_0731__1, NC_007_S_0068__1, NC_007_S_0070__1, NC_007_S_1206__1, NC_010_S_0067__1, NC_010_S_0419__1, NC_011_S_0005__1, NC_011_S_0021__1, NC_011_S_0023__1, NC_067_S_0056__1, NC_067_S_0056__2, NC_094_S_1241__1, NC_094_S_1241__2]

# exemplary creation of data frame from extracted features
def create_dataframe_from_features():
	df_whitesumsum_AD2 = pd.DataFrame()
	df_whitesumsum_AD2['whitesumsumAD2'] = get_sumsum(white_sum_AD2)
	df_whitesumsum_AD2.to_csv("whitesumsum_AD2.csv")

	df_whitesumsum_NC2 = pd.DataFrame()
	df_whitesumsum_NC2['whitesumsumNC2'] = get_sumsum(white_sum_CN2)
	df_whitesumsum_NC2.to_csv("whitesumsum_NC2.csv")

	df_graysumsum_AD2 = pd.DataFrame()
	df_graysumsum_AD2["graysumsum_AD2"] = get_sumsum(gray__sum_AD2)
	df_graysumsum_AD2.to_csv("graysumsum_AD2.csv")

	df_graysumsum_NC2 = pd.DataFrame()
	df_graysumsum_NC2["graysumsum_NC2"] = get_sumsum(gray_sum_CN2)
	df_graysumsum_NC2.to_csv("graysumsum_NC2.csv")

	df_csfsumsum_AD2 = pd.DataFrame()
	df_csfsumsum_AD2['csvsumsumAD2'] = get_sumsum(csf_sum_AD2)
	df_csfsumsum_AD2.to_csv("csfsumsum_AD2.csv")

	df_csfsumsum_NC2 = pd.DataFrame()
	df_csfsumsum_NC2['csfsumsumNC2'] = get_sumsum(csf_sum_CN2)
	df_csfsumsum_NC2.to_csv('csfsumsum_NC2.csv')





########################
########################
## Feature Extraction ##
########################
########################


# extracting features from time series of data in spiral drawings data set
def extract_features(df):
    
    XList = df['X'].tolist()
    XValues = [float(i) for i in XList]
    YList = df['Y'].tolist()
    YValues = [float(i) for i in YList]
    ZList = df['Z'].tolist()
    ZValues = [float(i) for i in ZList]
    PressList = df['Pressure'].tolist()
    PressValues = [float(i) for i in PressList]
    GripList = df['GripAngle'].tolist()
    GripValues = [float(i) for i in GripList]
    IDList = df['ID'].tolist()
    IDValues = [float(i) for i in IDList]
    
    ID = IDList[1]
    
    feature1Z = mean(ZValues)
    feature1Y = mean(YValues)
    feature1X = mean(XValues)
    feature1P = mean(PressValues)
    feature1G = mean(GripValues)
    
    feature2X = np.std(XValues, axis=0)
    feature2Y = np.std(YValues, axis=0)
    feature2Z = np.std(ZValues, axis=0)
    feature2P = np.std(PressValues, axis=0)
    feature2G = np.std(GripValues, axis=0)
    
    vZ = [abs(x[1]-x[0]) for x in zip(ZValues[1:], ZValues[:-1])]
    summeZ = sum(vZ)
    vX = [abs(x[1]-x[0]) for x in zip(XValues[1:], XValues[:-1])]
    summeX = sum(vX)
    vY = [abs(x[1]-x[0]) for x in zip(YValues[1:], YValues[:-1])]
    summeY = sum(vY)
    vP = [abs(x[1]-x[0]) for x in zip(PressValues[1:], PressValues[:-1])]
    summeP = sum(vP)
    vG = [abs(x[1]-x[0]) for x in zip(GripValues[1:], GripValues[:-1])]
    summeG = sum(vG)
    
    feature3Z = summeZ/(len(ZValues)-1)
    feature3X = summeX/(len(XValues)-1)
    feature3Y = summeY/(len(YValues)-1)
    feature3P = summeP/(len(PressValues) -1)
    feature3G = summeG/(len(GripValues) -1)
    
    wX = [abs(x[1]-x[0]) for x in zip(XValues[2:],XValues[:-2])]
    summeWX = sum(wX)
    wY = [abs(x[1]-x[0]) for x in zip(YValues[2:],YValues[:-2])]
    summeWY = sum(wY)
    wZ = [abs(x[1]-x[0]) for x in zip(ZValues[2:],ZValues[:-2])]
    summeWZ = sum(wZ)
    wP = [abs(x[1]-x[0]) for x in zip(PressValues[2:],PressValues[:-2])]
    summeWP = sum(wP)
    wG = [abs(x[1]-x[0]) for x in zip(GripValues[2:],GripValues[:-2])]
    summeWG = sum(wG)
    
    feature4X = summeWX/(len(XValues)-1)
    feature4Y = summeWY/(len(YValues)-1)
    feature4Z = summeWZ/(len(ZValues)-1)
    feature4P = summeWP/(len(PressValues)-1)
    feature4G = summeWG/(len(GripValues)-1)

    feature5X = (feature3X/feature2X)
    feature5Y = (feature3Y/feature2Y)
    feature5Z = (feature3Z/feature2Z)
    feature5P = (feature3P/feature2P)
    feature5G = (feature4G/feature2G)

    feature6X = (feature5X/feature2X)
    feature6Y = (feature5Y/feature2Y)
    feature6Z = (feature5Z/feature2Z)
    feature6P = (feature5P/feature2P)
    feature6G = (feature5G/feature2G)
    
    feature7Z = median(ZValues)
    feature7Y = median(YValues)
    feature7X = median(XValues)
    feature7P = median(PressValues)
    feature7G = median(GripValues)

    feature8Z = variance(ZValues)
    feature8Y = variance(YValues)
    feature8X = variance(XValues)
    feature8P = variance(PressValues)
    feature8G = variance(GripValues)
    
    list1 = [feature1X,feature1Y,feature1Z, feature1P,feature1G,
         feature2X,feature2Y,feature2Z, feature2P,feature2G,
         feature3X,feature3Y,feature3Z, feature3P,feature3G,
         feature4X,feature4Y,feature4Z, feature4P,feature4G,
         feature5X,feature5Y,feature5Z, feature5P,feature5G,
         feature6X,feature6Y,feature6Z, feature6P,feature6G,
         feature7X,feature7Y,feature7Z, feature7P,feature7G,
         feature8X,feature8Y,feature8Z, feature8P,feature8G,
         df.shape[0],
         ID]
    
    return list1


row0 = extract_features(df0)
row1 = extract_features(df1)
row2 = extract_features(df2)
row3 = extract_features(df3)
row4 = extract_features(df4)
row5 = extract_features(df5)
row6 = extract_features(df6)
row7 = extract_features(df7)
row8 = extract_features(df8)
row9 = extract_features(df9)
row10 = extract_features(df10)
row11 = extract_features(df11)
row12 = extract_features(df12)
row13 = extract_features(df13)
row14 = extract_features(df14)
row15 = extract_features(df15)
row16 = extract_features(df16)
row17 = extract_features(df17)
row18 = extract_features(df18)
row19 = extract_features(df19)
row20 = extract_features(df20)
row21 = extract_features(df21)
row22 = extract_features(df22)
row23 = extract_features(df23)
row24 = extract_features(df24)

dataframe = pd.DataFrame([row0,row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12,row13,row14,row15,row16,row17,row18,row19,row20,row21,row22,row23,row24],
                         columns=['feature1X','feature1Y','feature1Z','feature1P','feature1G','feature2X','feature2Y','feature2Z', 'feature2P','feature2G',
         'feature3X','feature3Y','feature3Z', 'feature3P','feature3G',
         'feature4X','feature4Y','feature4Z', 'feature4P','feature4G',
         'feature5X','feature5Y','feature5Z', 'feature5P','feature5G',
         'feature6X','feature6Y','feature6Z', 'feature6P','feature6G',
         'feature7X','feature7Y','feature7Z', 'feature7P','feature7G',
         'feature8X','feature8Y','feature8Z', 'feature8P','feature8G',
         'length',
         'ID'])


###################
###################
## LOSO approach ##
###################
###################


def build_lists_data():
    global list_X_trains, list_X_tests, list_y_tests, list_y_trains
    list_X_trains = [X_train_wo1, X_train_wo2, X_train_wo4, X_train_wo5, X_train_wo6, X_train_wo7, X_train_wo8, X_train_wo10, X_train_wo13,X_train_wo16,X_train_wo17,X_train_wo18,X_train_wo19,X_train_wo20,X_train_wo21,X_train_wo22,X_train_wo24,X_train_wo25,X_train_wo26,X_train_wo27,X_train_wo31,X_train_wo32,X_train_wo33,X_train_wo34,X_train_wo35,X_train_wo37,X_train_wo39,X_train_wo42,X_train_wo43,X_train_wo44,X_train_wo49,X_train_wo50]
    list_X_tests = [X_test_only1, X_test_only2, X_test_only4, X_test_only5, X_test_only6, X_test_only7, X_test_only8, X_test_only10, X_test_only13,X_test_only16,X_test_only17,X_test_only18,X_test_only19,X_test_only20,X_test_only21,X_test_only22,X_test_only24,X_test_only25,X_test_only26,X_test_only27,X_test_only31,X_test_only32,X_test_only33,X_test_only34,X_test_only35,X_test_only37,X_test_only39,X_test_only42,X_test_only43,X_test_only44,X_test_only49,X_test_only50]
    list_y_tests = [y_test_only1, y_test_only2, y_test_only4, y_test_only5, y_test_only6, y_test_only7, y_test_only8, y_test_only10, y_test_only13,y_test_only16,y_test_only17,y_test_only18,y_test_only19,y_test_only20,y_test_only21,y_test_only22,y_test_only24,y_test_only25,y_test_only26,y_test_only27,y_test_only31,y_test_only32,y_test_only33,y_test_only34,y_test_only35,y_test_only37,y_test_only39,y_test_only42,y_test_only43,y_test_only44,y_test_only49,y_test_only50]
    list_y_trains = [y_train_wo1, y_train_wo2, y_train_wo4, y_train_wo5, y_train_wo6, y_train_wo7, y_train_wo8, y_train_wo10, y_train_wo13,y_train_wo16,y_train_wo17,y_train_wo18,y_train_wo19,y_train_wo20,y_train_wo21,y_train_wo22,y_train_wo24,y_train_wo25,y_train_wo26,y_train_wo27,y_train_wo31,y_train_wo32,y_train_wo33,y_train_wo34,y_train_wo35,y_train_wo37,y_train_wo39,y_train_wo42,y_train_wo43,y_train_wo44,y_train_wo49,y_train_wo50]



# creating several random forests for testing LOSO
def randomForest(criterion, n, maxdepth):
    y_pred1 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo1, y_train_wo1).score(X_test_only1, y_test_only1)
    print("Participant 1: ", y_pred1)
    
    y_pred2 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo2, y_train_wo2).score(X_test_only2, y_test_only2)
    print("Participant 2: ", y_pred2)

    y_pred4 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo4, y_train_wo4).score(X_test_only4, y_test_only4)
    print("Participant 4: ", y_pred4)

    y_pred5 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo5, y_train_wo5).score(X_test_only5, y_test_only5)
    print("Participant 5: ", y_pred5)

    y_pred6 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo6, y_train_wo6).score(X_test_only6, y_test_only6)
    print("Participant 6: ", y_pred6)

    y_pred7 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo7, y_train_wo7).score(X_test_only7, y_test_only7)
    print("Participant 7: ", y_pred7)

    y_pred8 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo8, y_train_wo8).score(X_test_only8, y_test_only8)
    print("Participant 8: ", y_pred8)

    y_pred10 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo10, y_train_wo10).score(X_test_only10, y_test_only10)
    print("Participant 10: ", y_pred10)

    y_pred13 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo13, y_train_wo13).score(X_test_only13, y_test_only13)
    print("Participant 13: ", y_pred13)

    y_pred16 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo16, y_train_wo16).score(X_test_only16, y_test_only16)
    print("Participant 16: ", y_pred16)

    y_pred17 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo17, y_train_wo17).score(X_test_only17, y_test_only17)
    print("Participant 17: ", y_pred17)

    y_pred18 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo18, y_train_wo18).score(X_test_only18, y_test_only18)
    print("Participant 18: ", y_pred18)

    y_pred19 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo19, y_train_wo19).score(X_test_only19, y_test_only19)
    print("Participant 19: ", y_pred19)

    y_pred20 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo20, y_train_wo20).score(X_test_only20, y_test_only20)
    print("Participant 20: ", y_pred20)

    y_pred21 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo21, y_train_wo21).score(X_test_only21, y_test_only21)
    print("Participant 21: ", y_pred21)

    y_pred22 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo22, y_train_wo22).score(X_test_only22, y_test_only22)
    print("Participant 22: ", y_pred22)
    
    y_pred24 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo24, y_train_wo24).score(X_test_only24, y_test_only24)
    print("Participant 24: ", y_pred24)

    y_pred25 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo25, y_train_wo25).score(X_test_only25, y_test_only25)
    print("Participant 25: ", y_pred25)

    y_pred26 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo26, y_train_wo26).score(X_test_only26, y_test_only26)
    print("Participant 26: ", y_pred26)

    y_pred27 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo27, y_train_wo27).score(X_test_only27, y_test_only27)
    print("Participant 27: ", y_pred27)

    y_pred31 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo31, y_train_wo31).score(X_test_only31, y_test_only31)
    print("Participant 31: ", y_pred31)

    y_pred32 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo32, y_train_wo32).score(X_test_only32, y_test_only32)
    print("Participant 32: ", y_pred32)

    y_pred33 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo33, y_train_wo33).score(X_test_only33, y_test_only33)
    print("Participant 33: ", y_pred33)

    y_pred34 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo34, y_train_wo34).score(X_test_only34, y_test_only34)
    print("Participant 34: ", y_pred34)

    y_pred35 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo35, y_train_wo35).score(X_test_only35, y_test_only35)
    print("Participant 35: ", y_pred35)

    y_pred37 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo37, y_train_wo37).score(X_test_only37, y_test_only37)
    print("Participant 37: ", y_pred37)

    y_pred39 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo39, y_train_wo39).score(X_test_only39, y_test_only39)
    print("Participant 39: ", y_pred39)

    y_pred42 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo42, y_train_wo42).score(X_test_only42, y_test_only42)
    print("Participant 42: ", y_pred42)

    y_pred43 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo43, y_train_wo43).score(X_test_only43, y_test_only43)
    print("Participant 43: ", y_pred43)

    y_pred44 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo44, y_train_wo44).score(X_test_only44, y_test_only44)
    print("Participant 44: ", y_pred44)

    y_pred49 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo49, y_train_wo49).score(X_test_only49, y_test_only49)
    print("Participant 49: ", y_pred49)

    y_pred50 = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=maxdepth).fit(X_train_wo50, y_train_wo50).score(X_test_only50, y_test_only50)
    print("Participant 50: ", y_pred50)


    res = y_pred1 + y_pred2 + y_pred4 + y_pred5 + y_pred6 + y_pred7 + y_pred8 + y_pred10 + y_pred13 + y_pred16 + y_pred17 + y_pred18 + y_pred19 + y_pred20 + y_pred21 + y_pred22 + y_pred24 + y_pred25 + y_pred26 + y_pred27 + y_pred31 + y_pred32 + y_pred33 + y_pred34 + y_pred35 + y_pred37 + y_pred39 + y_pred42 + y_pred43 + y_pred44 + y_pred49 + y_pred50
    print(res/24)



# separate files for LOSO approach 
# one participant always has to be deleted from the rest and stored separately as the test set
def create_separate_files(name):
    df = create_dataframe(name)
    
    for i in range(1, 51):
        l=[]
        tmp = df.copy()
        df2 = pd.DataFrame()

        if (i < 10):
            st = "S0"+str(i)
        else :
            st = "S" + str(i)
        for j in range (0, 195) :
            if(st in tmp['name'][j]) :
                a = tmp.iloc[j]
                df2 = df2.append(a)
                l.append(j)
                
        for x in l:
            tmp = tmp.drop(x)

        if not df2.empty:
            df2.to_csv("Status_only" + str(i) + ".csv", sep=',', index=False)
            tmp.to_csv("Status_without" + str(i) + ".csv", sep=',', index=False)

# use LOSO approach
def LOSO_result_2(list_Xtrains, list_ytrains, list_Xtests, list_ytests):
    # 1. check lists have same lenghts
    if(not(len(list_Xtrains) == len(list_ytests) == len(list_Xtests) == len(list_ytrains) )):
        return -1
       
    # 2. create result list in which all accuracies will be stored
    tmp = []
       
    # 3. loop through input lists and calculate accuracies for each LOSO iteration
    for i in range(0, len(list_Xtrains)):
        tmp.append(get_accuracy_reduced_2(Xtrain=list_Xtrains[i], ytrain=list_ytrains[i], Xtest=list_Xtests[i], ytest=list_ytests[i]))
    print("tmp: {}".format(tmp))

    # 4. return overall accuracy for whole LOSO implementation
    return (sum(tmp)/len(tmp))



def LOSO_result(list_Xtrains, list_ytrains, list_Xtests, list_ytests):
    # 1. check lists have same lenghts
    if(not(len(list_Xtrains) == len(list_ytests) == len(list_Xtests) == len(list_ytrains) )):
        return -1
       
    # 2. create result list in which all accuracies will be stored
    tmp = []
       
    # 3. loop through input lists and calculate accuracies for each LOSO iteration
    for i in range(0, len(list_Xtrains)):
        tmp.append(get_accuracy(Xtrain=list_Xtrains[i], ytrain=list_ytrains[i], Xtest=list_Xtests[i], ytest=list_ytests[i]))
    #print("tmp: {}".format(tmp))
    #print("sum: {}".format(sum(tmp)))
    #print("len: {}".format(len(tmp)))
    #print("res: {}".format(sum(tmp)/len(tmp)))
    # 4. return overall accuracy for whole LOSO implementation
    return (sum(tmp)/len(tmp))


def set_ground_truth():
    groundTruth = create_separate_files_Status("data.csv")
    X_train_wo1 = pd.read_csv("Features_woStatus_without1.csv", sep = ',')
    X_test_only1 = pd.read_csv("Features_woStatus_only1.csv", sep = ',')
    y_train_wo1 = (pd.read_csv("Status_without1.csv", sep = ',')['Status']).tolist()
    y_test_only1 = (pd.read_csv("Status_only1.csv", sep = ',')['Status']).tolist()


# Some variables for the LOSO approach
groundTruth = create_separate_files_Status("data.csv");
X_train_wo1 = pd.read_csv("Features_woStatus_without1.csv", sep = ',');
del X_train_wo1["name"];
X_test_only1 = pd.read_csv("Features_woStatus_only1.csv", sep = ',');
del X_test_only1["name"];
y_train_wo1 = (pd.read_csv("Status_without1.csv", sep = ',')['Status']).tolist();
y_test_only1 = (pd.read_csv("Status_only1.csv", sep = ',')['Status']).tolist();

X_train_wo2 = pd.read_csv("Features_woStatus_without2.csv", sep = ',');
del X_train_wo2["name"];
X_test_only2 = pd.read_csv("Features_woStatus_only2.csv", sep = ',');
del X_test_only2["name"];
y_train_wo2 = (pd.read_csv("Status_without2.csv", sep = ',')['Status']).tolist();
y_test_only2 = (pd.read_csv("Status_only2.csv", sep = ',')['Status']).tolist();

X_train_wo4 = pd.read_csv("Features_woStatus_without4.csv", sep = ',');
del X_train_wo4["name"];
X_test_only4 = pd.read_csv("Features_woStatus_only4.csv", sep = ',');
del X_test_only4["name"];
y_train_wo4 = (pd.read_csv("Status_without4.csv", sep = ',')['Status']).tolist();
y_test_only4 = (pd.read_csv("Status_only4.csv", sep = ',')['Status']).tolist();

X_train_wo5 = pd.read_csv("Features_woStatus_without5.csv", sep = ',');
del X_train_wo5["name"];
X_test_only5 = pd.read_csv("Features_woStatus_only5.csv", sep = ',');
del X_test_only5["name"];
y_train_wo5 = (pd.read_csv("Status_without5.csv", sep = ',')['Status']).tolist();
y_test_only5 = (pd.read_csv("Status_only5.csv", sep = ',')['Status']).tolist();

X_train_wo6 = pd.read_csv("Features_woStatus_without6.csv", sep = ',');
del X_train_wo6["name"];
X_test_only6 = pd.read_csv("Features_woStatus_only6.csv", sep = ',');
del X_test_only6["name"];
y_train_wo6 = (pd.read_csv("Status_without6.csv", sep = ',')['Status']).tolist();
y_test_only6 = (pd.read_csv("Status_only6.csv", sep = ',')['Status']).tolist();

X_train_wo7 = pd.read_csv("Features_woStatus_without7.csv", sep = ',');
del X_train_wo7["name"];
X_test_only7 = pd.read_csv("Features_woStatus_only7.csv", sep = ',');
del X_test_only7["name"];
y_train_wo7 = (pd.read_csv("Status_without7.csv", sep = ',')['Status']).tolist();
y_test_only7 = (pd.read_csv("Status_only7.csv", sep = ',')['Status']).tolist();

X_train_wo8 = pd.read_csv("Features_woStatus_without8.csv", sep = ',');
del X_train_wo8["name"];
X_test_only8 = pd.read_csv("Features_woStatus_only8.csv", sep = ',');
del X_test_only8["name"];
y_train_wo8 = (pd.read_csv("Status_without8.csv", sep = ',')['Status']).tolist();
y_test_only8 = (pd.read_csv("Status_only8.csv", sep = ',')['Status']).tolist();

X_train_wo10 = pd.read_csv("Features_woStatus_without10.csv", sep = ',');
del X_train_wo10["name"];
X_test_only10 = pd.read_csv("Features_woStatus_only10.csv", sep = ',');
del X_test_only10["name"];
y_train_wo10 = (pd.read_csv("Status_without10.csv", sep = ',')['Status']).tolist();
y_test_only10 = (pd.read_csv("Status_only10.csv", sep = ',')['Status']).tolist();

X_train_wo13 = pd.read_csv("Features_woStatus_without13.csv", sep = ',');
del X_train_wo13["name"];
X_test_only13 = pd.read_csv("Features_woStatus_only13.csv", sep = ',');
del X_test_only13["name"];
y_train_wo13 = (pd.read_csv("Status_without13.csv", sep = ',')['Status']).tolist();
y_test_only13 = (pd.read_csv("Status_only13.csv", sep = ',')['Status']).tolist();

X_train_wo16 = pd.read_csv("Features_woStatus_without16.csv", sep = ',');
del X_train_wo16["name"];
X_test_only16 = pd.read_csv("Features_woStatus_only16.csv", sep = ',');
del X_test_only16["name"];
y_train_wo16 = (pd.read_csv("Status_without16.csv", sep = ',')['Status']).tolist();
y_test_only16 = (pd.read_csv("Status_only16.csv", sep = ',')['Status']).tolist();

X_train_wo17 = pd.read_csv("Features_woStatus_without17.csv", sep = ',');
del X_train_wo17["name"];
X_test_only17 = pd.read_csv("Features_woStatus_only17.csv", sep = ',');
del X_test_only17["name"];
y_train_wo17 = (pd.read_csv("Status_without17.csv", sep = ',')['Status']).tolist();
y_test_only17 = (pd.read_csv("Status_only17.csv", sep = ',')['Status']).tolist();

X_train_wo18 = pd.read_csv("Features_woStatus_without18.csv", sep = ',');
del X_train_wo18["name"];
X_test_only18 = pd.read_csv("Features_woStatus_only18.csv", sep = ',');
del X_test_only18["name"];
y_train_wo18 = (pd.read_csv("Status_without18.csv", sep = ',')['Status']).tolist();
y_test_only18 = (pd.read_csv("Status_only18.csv", sep = ',')['Status']).tolist();

X_train_wo19 = pd.read_csv("Features_woStatus_without19.csv", sep = ',');
del X_train_wo19["name"];
X_test_only19 = pd.read_csv("Features_woStatus_only19.csv", sep = ',');
del X_test_only19["name"];
y_train_wo19 = (pd.read_csv("Status_without19.csv", sep = ',')['Status']).tolist();
y_test_only19 = (pd.read_csv("Status_only19.csv", sep = ',')['Status']).tolist();

X_train_wo20 = pd.read_csv("Features_woStatus_without20.csv", sep = ',');
del X_train_wo20["name"];
X_test_only20 = pd.read_csv("Features_woStatus_only20.csv", sep = ',');
del X_test_only20["name"];
y_train_wo20 = (pd.read_csv("Status_without20.csv", sep = ',')['Status']).tolist();
y_test_only20 = (pd.read_csv("Status_only20.csv", sep = ',')['Status']).tolist();

X_train_wo21 = pd.read_csv("Features_woStatus_without21.csv", sep = ',');
del X_train_wo21["name"];
X_test_only21 = pd.read_csv("Features_woStatus_only21.csv", sep = ',');
del X_test_only21["name"];
y_train_wo21 = (pd.read_csv("Status_without21.csv", sep = ',')['Status']).tolist();
y_test_only21 = (pd.read_csv("Status_only21.csv", sep = ',')['Status']).tolist();

X_train_wo22 = pd.read_csv("Features_woStatus_without22.csv", sep = ',');
del X_train_wo22["name"];
X_test_only22 = pd.read_csv("Features_woStatus_only22.csv", sep = ',');
del X_test_only22["name"];
y_train_wo22 = (pd.read_csv("Status_without22.csv", sep = ',')['Status']).tolist();
y_test_only22 = (pd.read_csv("Status_only22.csv", sep = ',')['Status']).tolist();

X_train_wo24 = pd.read_csv("Features_woStatus_without24.csv", sep = ',');
del X_train_wo24["name"];
X_test_only24 = pd.read_csv("Features_woStatus_only24.csv", sep = ',');
del X_test_only24["name"];
y_train_wo24 = (pd.read_csv("Status_without24.csv", sep = ',')['Status']).tolist();
y_test_only24 = (pd.read_csv("Status_only24.csv", sep = ',')['Status']).tolist();

X_train_wo25 = pd.read_csv("Features_woStatus_without25.csv", sep = ',');
del X_train_wo25["name"];
X_test_only25 = pd.read_csv("Features_woStatus_only25.csv", sep = ',');
del X_test_only25["name"];
y_train_wo25 = (pd.read_csv("Status_without25.csv", sep = ',')['Status']).tolist();
y_test_only25 = (pd.read_csv("Status_only25.csv", sep = ',')['Status']).tolist();

X_train_wo26 = pd.read_csv("Features_woStatus_without26.csv", sep = ',');
del X_train_wo26["name"];
X_test_only26 = pd.read_csv("Features_woStatus_only26.csv", sep = ',');
del X_test_only26["name"];
y_train_wo26 = (pd.read_csv("Status_without26.csv", sep = ',')['Status']).tolist();
y_test_only26 = (pd.read_csv("Status_only26.csv", sep = ',')['Status']).tolist();

X_train_wo27 = pd.read_csv("Features_woStatus_without27.csv", sep = ',');
del X_train_wo27["name"];
X_test_only27 = pd.read_csv("Features_woStatus_only27.csv", sep = ',');
del X_test_only27["name"];
y_train_wo27 = (pd.read_csv("Status_without27.csv", sep = ',')['Status']).tolist();
y_test_only27 = (pd.read_csv("Status_only27.csv", sep = ',')['Status']).tolist();

X_train_wo31 = pd.read_csv("Features_woStatus_without31.csv", sep = ',');
del X_train_wo31["name"];
X_test_only31 = pd.read_csv("Features_woStatus_only31.csv", sep = ',');
del X_test_only31["name"];
y_train_wo31 = (pd.read_csv("Status_without31.csv", sep = ',')['Status']).tolist();
y_test_only31 = (pd.read_csv("Status_only31.csv", sep = ',')['Status']).tolist();

X_train_wo32 = pd.read_csv("Features_woStatus_without32.csv", sep = ',');
del X_train_wo32["name"];
X_test_only32 = pd.read_csv("Features_woStatus_only32.csv", sep = ',');
del X_test_only32["name"];
y_train_wo32 = (pd.read_csv("Status_without32.csv", sep = ',')['Status']).tolist();
y_test_only32 = (pd.read_csv("Status_only32.csv", sep = ',')['Status']).tolist();

X_train_wo33 = pd.read_csv("Features_woStatus_without33.csv", sep = ',');
del X_train_wo33["name"];
X_test_only33 = pd.read_csv("Features_woStatus_only33.csv", sep = ',');
del X_test_only33["name"];
y_train_wo33 = (pd.read_csv("Status_without33.csv", sep = ',')['Status']).tolist();
y_test_only33 = (pd.read_csv("Status_only33.csv", sep = ',')['Status']).tolist();

X_train_wo34 = pd.read_csv("Features_woStatus_without34.csv", sep = ',');
del X_train_wo34["name"];
X_test_only34 = pd.read_csv("Features_woStatus_only34.csv", sep = ',');
del X_test_only34["name"];
y_train_wo34 = (pd.read_csv("Status_without34.csv", sep = ',')['Status']).tolist();
y_test_only34 = (pd.read_csv("Status_only34.csv", sep = ',')['Status']).tolist();

X_train_wo35 = pd.read_csv("Features_woStatus_without35.csv", sep = ',');
del X_train_wo35["name"];
X_test_only35 = pd.read_csv("Features_woStatus_only35.csv", sep = ',');
del X_test_only35["name"];
y_train_wo35 = (pd.read_csv("Status_without35.csv", sep = ',')['Status']).tolist();
y_test_only35 = (pd.read_csv("Status_only35.csv", sep = ',')['Status']).tolist();

X_train_wo37 = pd.read_csv("Features_woStatus_without37.csv", sep = ',');
del X_train_wo37["name"];
X_test_only37 = pd.read_csv("Features_woStatus_only37.csv", sep = ',');
del X_test_only37["name"];
y_train_wo37 = (pd.read_csv("Status_without37.csv", sep = ',')['Status']).tolist();
y_test_only37 = (pd.read_csv("Status_only37.csv", sep = ',')['Status']).tolist();

X_train_wo39 = pd.read_csv("Features_woStatus_without39.csv", sep = ',');
del X_train_wo39["name"];
X_test_only39 = pd.read_csv("Features_woStatus_only39.csv", sep = ',');
del X_test_only39["name"];
y_train_wo39 = (pd.read_csv("Status_without39.csv", sep = ',')['Status']).tolist();
y_test_only39 = (pd.read_csv("Status_only39.csv", sep = ',')['Status']).tolist();

X_train_wo42 = pd.read_csv("Features_woStatus_without42.csv", sep = ',');
del X_train_wo42["name"];
X_test_only42 = pd.read_csv("Features_woStatus_only42.csv", sep = ',');
del X_test_only42["name"];
y_train_wo42 = (pd.read_csv("Status_without42.csv", sep = ',')['Status']).tolist();
y_test_only42 = (pd.read_csv("Status_only42.csv", sep = ',')['Status']).tolist();

X_train_wo43 = pd.read_csv("Features_woStatus_without43.csv", sep = ',');
del X_train_wo43["name"];
X_test_only43 = pd.read_csv("Features_woStatus_only43.csv", sep = ',');
del X_test_only43["name"];
y_train_wo43 = (pd.read_csv("Status_without43.csv", sep = ',')['Status']).tolist();
y_test_only43 = (pd.read_csv("Status_only43.csv", sep = ',')['Status']).tolist();

X_train_wo44 = pd.read_csv("Features_woStatus_without44.csv", sep = ',');
del X_train_wo44["name"];
X_test_only44 = pd.read_csv("Features_woStatus_only44.csv", sep = ',');
del X_test_only44["name"];
y_train_wo44 = (pd.read_csv("Status_without44.csv", sep = ',')['Status']).tolist();
y_test_only44 = (pd.read_csv("Status_only44.csv", sep = ',')['Status']).tolist();

X_train_wo49 = pd.read_csv("Features_woStatus_without49.csv", sep = ',');
del X_train_wo49["name"];
X_test_only49 = pd.read_csv("Features_woStatus_only49.csv", sep = ',');
del X_test_only49["name"];
y_train_wo49 = (pd.read_csv("Status_without49.csv", sep = ',')['Status']).tolist();
y_test_only49 = (pd.read_csv("Status_only49.csv", sep = ',')['Status']).tolist();

X_train_wo50 = pd.read_csv("Features_woStatus_without50.csv", sep = ',');
del X_train_wo50["name"];
X_test_only50 = pd.read_csv("Features_woStatus_only50.csv", sep = ',');
del X_test_only50["name"];
y_train_wo50 = (pd.read_csv("Status_without50.csv", sep = ',')['Status']).tolist();
y_test_only50 = (pd.read_csv("Status_only50.csv", sep = ',')['Status']).tolist();

y_test_only1=[1,1,1,1,1,1];
y_test_only2 = [1,1,1,1,1,1];
y_test_only4 = [1,1,1,1,1,1];
y_test_only5 = [1,1,1,1,1,1];
y_test_only6 = [1,1,1,1,1,1];
y_test_only7 = [0,0,0,0,0,0];
y_test_only8 = [1,1,1,1,1,1];
y_test_only10= [0,0,0,0,0,0];
y_test_only13= [0,0,0,0,0,0];
y_test_only16= [1,1,1,1,1,1];
y_test_only17= [0,0,0,0,0,0];
y_test_only18= [1,1,1,1,1,1];
y_test_only19= [1,1,1,1,1,1];
y_test_only20= [1,1,1,1,1,1];
y_test_only21= [1,1,1,1,1,1,1];
y_test_only22= [1,1,1,1,1,1];
y_test_only24= [1,1,1,1,1,1];
y_test_only25= [1,1,1,1,1,1];
y_test_only26= [1,1,1,1,1,1];
y_test_only27= [1,1,1,1,1,1,1];
y_test_only31= [1,1,1,1,1,1];
y_test_only32= [1,1,1,1,1,1];
y_test_only33= [1,1,1,1,1,1];
y_test_only34= [1,1,1,1,1,1];
y_test_only35= [1,1,1,1,1,1,1];
y_test_only37= [1,1,1,1,1,1];
y_test_only39= [1,1,1,1,1,1];
y_test_only42= [0,0,0,0,0,0];
y_test_only43= [0,0,0,0,0,0];
y_test_only44= [1,1,1,1,1,1];
y_test_only49= [0,0,0,0,0,0];
y_test_only50= [0,0,0,0,0,0];