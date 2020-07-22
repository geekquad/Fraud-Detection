# Fraud-Detection
The Enron fraud is the largest case of corporate fraud in American history. Founded in 1985, Enron Corporation went bankrupt by end of 2001 due to widespread corporate fraud and corruption. Before its fall, Fortune magazine had named Enron “America’s most innovative company” for six consecutive years.

**Dataset**: <a href="https://www.cs.cmu.edu/~./enron/"> https://www.cs.cmu.edu/~./enron/ </a> 
<hr></hr>

## <u> Goal of the Porject: </u>
The goal of the project is to go through the thought process of data exploration (learning, cleaning and preparing the data), 
feature selecting/engineering (selecting the features which influence mostly on the target, 
create new features (which explains the target the better than existing) and, 
reducing the dimensionality of the data using principal component analysis (PCA)), 
picking/tuning one of the supervised machine learning algorithm and validating it to get the accurate person of interest identifier model.

## <u> Data Exploration </u>
The features in the data fall into three major types, namely 
- financial features, 
- email features 
- POI labels.

There are 143 samples with 20 features and a binary classification ("poi")
Among 146 samples, there are
- 18 POI and 
- 128 non-POI.

<hr> </hr>

## Optimize Feature Selection/Engineering
During the work on the project, I've played with the different features and models. One strategy was to standardize features, 
apply principal component analysis and GaussianNB classifier, another strategy was to use decision tree classifier, incl. choosing the 
features with features importance attribute and tuning the model.
### Create new features
For both strategies I've tried to create new features as a fraction of almost all financial variables (f.ex. fractional bonus 
as fraction of bonus to total_payments, etc.). Logic behind email feature creation was to check the fraction of emails, sent to POI, 
to all sent emails; emails, received from POI, to all received emails.
I've end up with using one new feature fraction_to_POI.
<hr> </hr>

## <u> Pick and Tune an Algorithm: </u>
I've played with 7 machine learning algorithms:
- Naive Bayes (GaussianNB)
- SVC
- RandomForestClassifier
- ExtraTreesClassifier
- AdaBoostClassifier
- LogisticRegression
- SVC

### Comparing Classifiers based on cross-validation scores:
- 1st tier: SVC, RandomForestClassifier
- 2nd tier: GaussianNB, ExtraTreesClassifier, AdaBoostClassifier
- 3rd tier: Logistic Regression, LinearSVC

### Tuning the algorithm:
Bias-variance tradeoff is one of the key dilema in machine learning. High bias algorithms has no capacity to learn, high variance algorithms 
react poorly in case they didn't see such data before. Predictive model should be tuned to achieve compromise. The process of changing the parameteres of algorithms is 
algorithm tuning and it lets us find the golden mean and best result. If I don't tune the algorithm well, I don't get the best result I could.
Algorithm might be tuned manually by iteratively changing the parameteres and tracking the results. Or GridSearchCV might be used which makes this automatically.
I've tuned the parameteres of my decision tree classifier by sequentially tuning parameter by parameter and got the best F1 using these parameters
<hr> </hr>

## Validate and Evaluate
### Usage of Evaluation Metrics
In the project I've used F1 score as key measure of algorithms' accuracy. It considers both the precision and the recall of the test to compute the score.
Precision is the ability of the classifier not label as positive sample that is negative.
Recall is the ability of the classifier to find all positive samples.
The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.

### Validation Strategy
The validation is a process of model performance evaluation. Classic mistace is to use small data set for the model training or validate model on the same data set as train it.
There are a number of strategies to validate the model. One of them is to split the available data into train and test data another one is to perform a cross validation: process of splitting the data on k beans equal size; run learning experiments; repeat this operation number of times and take the average test result.


