# CS4210-Machine-Learning
Projects from CS4210

## Assignment 1
### decision_tree.py
Convert a data set to an ID3 decision tree by one hot encoding non-numeric values,
then using sklearn to build the decision tree and plot the final output

## Assignment 2
### decision_tree_2.py
Build decision trees from differently sized data sets in the same way as assignment 1,
but now test the accuracy of the trees' predictions on a test data set

### knn.py
Use sklearn to build a KNN classifier, then use leave one out cross validation to
calculate the error rate

### naive_bayes.py
Use sklearn to build a naive bayes classifier, and only output classifications of the
test data set if the confidence level is above the specified threshold

## Assignment 3
### bagging_random_forest.py
Using sklearn, build a single decision tree classifier, an ensemble classifier of decision
trees built from 20 bootstrap samples, and a random forest classifier with the same number
of estimators. Then compare the accuracy of the 3 classifiers on the same test data set

### svm.py
Use sklearn to build a support vector machine classifier, testing different combinations
of hyperparameter values to find which set of values produces the most accurate classifier

## Assignment 4
### deep_learning.py
Use TensorFlow to build a deep neural network, testing different combinations for the
number of hidden layers, number of neurons, and learning rate to find which set of values
produces the most accurate neural network. Finally print information for the best network

### perceptron.py
Similarly, use sklearn to build various single and multi-layered perceptrons, testing
different values for the learning rate looking for the most accurate perceptron

## Assignment 5
### association_rule_mining.py
Build an apriori compliant data frame from the provided supermarket data set, then use
mlxtend to find the frequent items meeting the minimumm support threshold, and the
association rules meeting the minimum confidence threshold. Finally print the
statistics for each rule

### clustering.py
Use sklearn to run k-means on a data set, testing different values of k to find which
value maximizes the silhouette coefficient. Then find the homogeneity score for the best
k-means cluster
