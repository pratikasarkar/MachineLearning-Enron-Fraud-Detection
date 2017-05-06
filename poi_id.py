#!/usr/bin/python

import sys
import pickle
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint
import numpy

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
'salary',
'bonus',
'bonus_salary_ratio',
'director_fees',
'deferral_payments',
'deferred_income',
'director_fees',
'exercised_stock_options',
'expenses',
'from_messages',
'from_poi_to_this_person',
'from_this_person_to_poi',
'loan_advances',
'long_term_incentive',
'fraction_from_this_person_to_poi',
'fraction_from_poi_to_this_person',
'other',
'restricted_stock',
'restricted_stock_deferred',
'shared_receipt_with_poi',
'to_messages',
'total_payments',
'total_stock_value'] # You will need to use more features


#features_list = ['poi',
#'salary',
#'bonus',
#'bonus_salary_ratio',
##'director_fees',
##'deferral_payments',
#'deferred_income',
##'director_fees',
#'exercised_stock_options',
##'expenses',
##'from_messages',
#'from_poi_to_this_person',
##'from_this_person_to_poi',
#'loan_advances',
#'long_term_incentive',
#'fraction_from_this_person_to_poi',
##'fraction_from_poi_to_this_person',
##'other',
#'restricted_stock',
##'restricted_stock_deferred',
#'shared_receipt_with_poi',
##'to_messages']
#'total_payments',
#'total_stock_value']

### Dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
	

#Total number of point allocated
print ("Total number of points in the data : ", len(data_dict))


#Number of POIs
count_POI = 0
for person in data_dict:
	if data_dict[person]['poi']:
		count_POI+=1
print ("number of POI : ",count_POI)

#Number of Non-POIs
count_non_POI = 0
for person in data_dict:
	if not data_dict[person]['poi']:
		count_non_POI+=1
print ("number of non POI : ",count_non_POI)

### Task 2: Remove outliers	
### TOTAL is a clear outlier value from the data, because for comparison between almost any two features we can find out one particular outlier.
### So to remove the outlier, we can use the pop() method 

data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

### Task 3: Create new feature(s)
### Bonus-salary ratio
for employee, features in data_dict.items():
	if features['bonus'] == "NaN" or features['salary'] == "NaN":
		features['bonus_salary_ratio'] = "NaN"
	else:
		features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])
### Created 2 new features 'fraction_from_poi_to_this_person' and 'fraction_from_this_person_to_poi'. These fractions will tell the proportion of people in 
###contact with the pois.
### The more likely a person being in contact with POI, the more likely is he of being POI himself.
for name in data_dict:
	if data_dict[name]['from_this_person_to_poi'] == "NaN" or data_dict[name]['from_messages'] == "NaN":
		data_dict[name]['fraction_from_this_person_to_poi'] = "NaN"
	else : 
		data_dict[name]['fraction_from_this_person_to_poi'] = float(data_dict[name]['from_this_person_to_poi'])/float(data_dict[name]['from_messages'])

	if data_dict[name]['from_poi_to_this_person'] == "NaN" or data_dict[name]['to_messages'] == "NaN":
		data_dict[name]['fraction_from_poi_to_this_person'] = "NaN"
	else : 
		data_dict[name]['fraction_from_poi_to_this_person'] = float(data_dict[name]['from_poi_to_this_person'])/float(data_dict[name]['to_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Feature selection
from sklearn.feature_selection import SelectKBest
select = SelectKBest()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



from sklearn.cross_validation import train_test_split
feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# # Provided to give you a starting point. Try a variety of classifiers.

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
nb = GaussianNB()

from sklearn import tree
dtc = tree.DecisionTreeClassifier()


# Load pipeline steps into list
steps = [
         # Feature selection
         ('feature_selection', select),
         
         # Classifier
         #('dtc', dtc)
          ('nb', nb)
         ]

# Create pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps)

# Parameters to try in grid search
parameters = dict(
                  feature_selection__k=[7], 
                  ##dtc__criterion=['gini', 'entropy'],
                  # dtc__splitter=['best', 'random'],
                  ##dtc__max_depth=[None, 1, 2, 3, 4],
                  ##dtc__min_samples_split=[2, 10, 25, 50],
                  # dtc__min_samples_leaf=[1, 2, 3, 4],
                  # dtc__min_weight_fraction_leaf=[0, 0.25, 0.5],
                  ##dtc__random_state=[42]
                  )



# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ### using our testing script. Check the tester.py script in the final project
# 	### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info: 
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


# Create, fit, and make predictions with grid search
gs = GridSearchCV(pipeline,
	              param_grid=parameters)
gs.fit(features_train, labels_train)
labels_predictions = gs.predict(features_test)

clf = gs.best_estimator_
print ("\n", "Best parameters are: ", gs.best_params_, "\n")

# Accuracy of the best selected classifier
print()
print("Accuracy : ",accuracy_score(labels_test,clf.predict(feature_test)))
print("Precision : ",precision_score(labels_test,clf.predict(feature_test)))
print("Recall : ",recall_score(labels_test,clf.predict(feature_test)))
print()

# Print features selected and their importances
features_selected=[features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
#scores = clf.named_steps['feature_selection'].scores_
#importances = clf.named_steps['dtc'].feature_importances_
#indices = np.argsort(importances)[::-1]
print ('', len(features_selected), " features selected and their importances:")
for i in range(len(features_selected)):
	print ("feature no. {}: {}".format(i+1,features_selected[i]))
#for i in range(len(features_selected)):
 #   print ("feature no. {}: {} ({}) ({})".format(i+1,features_selected[indices[i]],importances[indices[i]], scores[indices[i]]))
 
# Print classification report (focus on precision and recall)
report = classification_report( labels_test, labels_predictions )
print(report)

# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)