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
