# machinelearning<br>
POI from ENRON Data using Machine learning<br>
<br>
** Summarize the Goal of the project :**<br>
<br>
The goal of this project is to build an algorithm to identify a person of interest based on financial and email data made public as a result of the Enron scandal. The initial list of persons of interest in the fraud case is made from individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.<br>
<br>
Firstly, I included all the features and then selected only the 7 best features that represented better information about the data<br>
<br>
The total number of data points (people in the training and testing dataset) = 146<br>
POI = 18<br>
Non-POI - 128<br>
<br>
In the initial dataset there are 21 available features, I selected all the features initially for getting a hang on the dataset.<br>
<br>
There were two outliers in the data that weren't needed in this investigation : 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. I deleted them from the dataset to avoid droping the accuracy of the identifier<br>
<br>
These outliers were detected by ploting the data on several axis and reaching for the extreme values to check on them. For the rest of the outliers they seem to be valid data points for the top ranking official of ENRON<br>
<br>
** Features used in the identifier :**<br>
<br>
In addition to selecting the most important features on the data set I added the following additional features to the features list:<br>
<br>
bonus_salary_ratio = bonus / salary<br>
This ratio helped finding a relationship between salary and bonus offered to a person. If the bonus was unusually high, the ratio would be high which could be an indication of the value pointing to a POI.<br>
<br>
fraction_from_this_person_to_poi = from_this_person_to_poi/from_messages<br>
This ratio helped in finding the relationship between the variables 'from_this_person_to_poi' and 'from_messages'. If this ratio is high, it could be an indiaction that most of the mails from this person are to a POI. So even the person sending the messages could be considered as a person of interest.<br>
<br>
fraction_from_poi_to_this_person = from_poi_to_this_person/to_messages<br>
This ratio helped in finding the relationship between the variables 'from_poi_to_this_person' and 'to_messages'. If this ratio is high, it could be an indiaction that most of the mails  this person receives are from a POI. So even the person receiving the messages could be considered as a person of interest.<br>
<br>
I added the above created features to my original features list. Then used SelectKBest algorithm for automated feature selection.<br>
I used the following parameters for the decision tree classification algo :<br>
 1 . min_sample_split - k best algorithm selected 10 to be the best value for minimum sample split as using higher value resulted in lower precision and recall values.<br>
 2 . max_depth - k best algorithm selected 'None' to be the best value in combination with min_sample_split which seems to give the maximum accuracy among all others.<br>
 3 . criterion - k best algorithm selected 'entropy' to be the best parameter for the criteria to measure the quality of a split.<br>
<br>
But further using Naive Bayes Classifier I found that it gave me a better accuracy, precision as well as recall using 7 best features. These 7 features were found to the best by multiple iterations of running the naive bayes over different number of features.<br><br>
The 7 best features selected by the SelectKBest algorithm are the following :<br>
<br>
feature no. 1: salary<br>
feature no. 2: bonus<br>
feature no. 3: bonus_salary_ratio<br>
feature no. 4: exercised_stock_options<br>
feature no. 5: fraction_from_this_person_to_poi<br>
feature no. 6: shared_receipt_with_poi<br>
feature no. 7: total_stock_value<br>
<br>
Here 2 of the 3 new features added appear in the selection by SelectKBest algorithm which are 'bonus_salary_ratio' and 'fraction_from_this_person_to_poi'.<br>
We can see that there are 5 financial features and 2 email features selected by the SelectKBest algorithm. These features were selected as there seemed to be some strong relationships between them. Salary and bonus were plotted on a scatterplot which showed a strong relationship with some outliers. exercised_stock_options and total_stock_value have a very strong relationship with a high value of correlation coefficient. Also most of the outliers in this relationship are POI according to the scatterplot. To find the plots, please follow the link : https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard<br>
The fact that shared_receipt_with_poi and fraction_from_this_person_to_poi is included after using SelectKBest proved that these are crucial features, as they also slightly increased the precision and recall of the machine learning algorithm used in the analysis.<br>
<br>
** Pick and Tune an Algorithm :**<br>
<br>
After trying some algorithms I found that decision tree algorithm has the potential to be improved further. Without any tuning, Naive Bayes Classifier performed reasonably sufficient with precision & recall rate both larger than 0.3.<br>
By using a tuned decision tree classifier with Kbest features selection I got the following metrics : <br>
Pipeline(steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_classif at 0x00000000050E6AE8>)), ('dtc', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best'))])<br>
        Accuracy: 0.83573       Precision: 0.36590      Recall: 0.31650 F1: 0.33941     F2: 0.32528<br>
        Total predictions: 15000        True positives:  633    False positives: 1097   False negatives: 1367   True negatives: 11903<br>
This is good enough but the recall without StratifiedShuffleSplit for cross validations is about 0.2 which was not good enough to meet the specifications.<br>
The DecisionTree classifier was tuned by the following parameters : - K best features : representing how many best features from 2 to 23. classification criterion : either gini for impurity or entropy for information gain - DecisionTree splitter : either the best splits or best random splits.<br>
<br>
Parameter Tuning for decision tree classifer:<br>
All the parameter tuning was done using multiple iterations with multiple combinations and finally the best parameters were found.<br>
1. criterion - I was able to choose from 2 criterion i.e, 'gini' or 'entropy' to measure the quality of the split. Gini is for impurity whereas entropy is for information gain. Using entropy gave greater accuracy, precision and recall for decision tree algorithm. So entropy was used as a parameter as the value of information gain would be greater.<br>
2. min_sample_split - I used mutliple values for sample split like 2, 10 ,25 and 50. I got the following results using these different values.<br>
	For min_sample_split = 2, Accuracy :  0.837209302326 Precision :  0.25 Recall :  0.2<br>
	For min_sample_split = 10, Accuracy :  0.860465116279 Precision :  0.333333333333 Recall :  0.2<br>
	For min_sample_split = 20, Accuracy :  0.837209302326 Precision :  0.25 Recall :  0.2<br>
	For min_sample_split = 30, Accuracy :  0.790697674419 Precision :  0.0 Recall :  0.0<br>
Among these min_sample_split of 10 seems to give the best value in terms of accuracy as well as precision and recall.<br>
3. max_depth - Multiple values were used for max depth too. I got the following results using these different values.<br>
	For max_depth = None, Accuracy :  0.860465116279 Precision :  0.333333333333 Recall :  0.2 (min_sample_split = 10)<br>
	For max_depth = 1, Accuracy :  0.8837 Precision :  0.0 Recall :  0.0 (min_sample_split = 2)<br>
	For max_depth = 2, Accuracy :  0.837209302326 Precision :  0.25 Recall :  0.2 (min_sample_split = 2)<br>
	For max_depth = 3, Accuracy :  0.837209302326 Precision :  0.25 Recall :  0.2 (min_sample_split = 2)<br>
	For max_depth = 4, Accuracy :  0.860465116279 Precision :  0.333333333333 Recall :  0.2 (min_sample_split = 10)<br>
Among these max_depth of 'None' seems to give the best value in terms of accuracy as well as precision and recall.<br>
<br>
Although initially, the result is not as expected, I believe with further tuning of decision tree classifier we can come up with a much better result.<br>
<br>
By using the Gaussian Naive Bayes classifier I get :<br>
GaussianNB(priors=None) Accuracy: 0.84613  Precision: 0.40855  Recall: 0.34400  F1: 0.37351  F2: 0.35523<br>
Total predictions: 15000   True positives:  688   False positives:  996   False negatives: 1312   True negatives: 12004<br>
<br>
Tuning the parameters of an algorithm is looking for the combination of parameters that will give the best performance for that given algorithm. In my case the algorithm I chose to be tuned was the feature selection and the parameters for the decision tree classifier. Even though it was tuned to get the best performance, it did not meet the specifications of a precision and recall scores above 0.3<br>
<br>
** Feature Scaling **<br>
<br>
As far as the approach I was taking for finding the POIs I did not find any need of adding any feature scaling. I did try using MinMaxScalar but as it seemed to reduce the performance of the algorithm,so I preferred to keep feature scaling out of this context of identifying the POI for the Enron dataset.<br>
<br>
** Validation **<br>
<br>
In order to test a classification algorithm we'll have to keep a portion of the dataset for the tests, the most common mistake is training the classifier on the whole dataset, while this may give good results on the tests ( since the test data will be the one that it has been trained for) it won't be able to give good results on new data that would be added. This is the case where it is too good to be true. In this investigation I started by leaving a 30% of the dataset for the testing since for both of the algorithms implemented the metrics are completly irrational when using a smaller portion of data for testing<br>
<br>
** Metrics **<br>
<br>
For this assignment, I used precision & recall as 2 main evaluation metrics. The best performance belongs to naive bayes (precision: 0.408 & recall: 0.344) which is also the final model of choice, as naive bayes is also widely used in text classification, we can actually extend this model for email classification if needed.<br>
Precision refer to the ratio of true positive (predicted as POI) to the records that are actually POI while recall described ratio of true positives to people flagged as POI.<br>
With a precision score of 0.40, it tells us that if this model predicts 100 POIs, then the chance would be 40 people who are truely POIs and the rest 60 are innocent. On the other hand, with a recall score of 0.344, this model can find 35% of all real POIs in prediction. Due to the nature of the dataset, accuracy is not a good measurement as even if non-POI are all flagged, the accuracy score will yield that the model is a success.<br>
