import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/rahul/Documents/IMR/Data_Decision_Tree/")

FullRaw = pd.read_csv('Telecom_Churn.csv')

# Check for NAs
FullRaw.isnull().sum()

# Summarize the data
FullRaw_Summary = FullRaw.describe()

# Remove Customer Id columns
FullRaw.drop(['customerID'], axis = 1, inplace = True)


########################
# Manual recoding of "Dependent Variable"
########################

# We will manually convert our categorical dependent variable to numeric
FullRaw['Churn'] = np.where(FullRaw['Churn'] == 'Yes', 1, 0)



########################
# Dummy variable creation
########################

# Dummy variable creation
FullRaw2 = pd.get_dummies(FullRaw)
FullRaw2.shape

############################
# Sampling: Divide the data into Train and Testset
############################

from sklearn.model_selection import train_test_split
Train, Test = train_test_split(FullRaw2, train_size=0.7, random_state = 123)

########################
# Sampling into X and Y
########################

# Divide each dataset into Indep Vars and Dep var
depVar = 'Churn'
trainX = Train.drop(depVar, axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop(depVar, axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape

########################################
# Decision Tree Model
########################################

from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.pyplot import figure

M1 = DecisionTreeClassifier(random_state = 123).fit(trainX, trainY) # Indep, Dep


########################################
# Model Visualization
########################################

# Vizualization of DT

figure() # This is like windows() function in R

DT_Plot1 = plot_tree(M1, fontsize = 10, feature_names = trainX.columns, filled = True, class_names = ["0","1"])
# filled = True allows the method to use colors to indicate the majority of the class
# class_names = ["0","1"] allows the method to provide the dependent variable "MAJORITY" 
# class name for each of colored boxes


############################
# Prediction and Validation on Testset
############################

from sklearn.metrics import classification_report

# Prediction on Testset
Test_Pred = M1.predict(testX)

# Classification Model Validation
Confusion_Mat = pd.crosstab(testY, Test_Pred)
Confusion_Mat # R, C format (Actual = testY, Predicted = Test_Pred)

# Validation on Testset
print(classification_report(testY, Test_Pred)) # Actual, Predicted


############
# DT Model 2
############

# Build Model
M2 = DecisionTreeClassifier(random_state = 123, min_samples_leaf = 500).fit(trainX, trainY)

# Vizualization of DT
figure()

DT_Plot2 = plot_tree(M2, fontsize = 10, feature_names = trainX.columns, filled = True, class_names = ["0","1"])


# Prediction on Testset
Test_Pred = M2.predict(testX)

# Classification Model Validation
Confusion_Mat = pd.crosstab(testY, Test_Pred)
Confusion_Mat # R, C format

# Validation on Testset
print(classification_report(testY, Test_Pred)) # Actual, Predicted


########################################
# Random Forest
########################################

from sklearn.ensemble import RandomForestClassifier

M1_RF = RandomForestClassifier(random_state = 123).fit(trainX, trainY)

# Prediction on Testset
Test_Pred = M1_RF.predict(testX)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testY, Test_Pred) # R, C format
Confusion_Mat 

# Validation on Testset
print(classification_report(testY, Test_Pred)) # Actual, Predicted


# Variable importance
M1_RF.feature_importances_

Var_Importance_Df = pd.concat([pd.DataFrame(M1_RF.feature_importances_),
                               pd.DataFrame(trainX.columns)], axis = 1)

Var_Importance_Df
Var_Importance_Df.columns = ["Value", "Variable_Name"]
Var_Importance_Df.sort_values("Value", ascending = False, inplace = True)
Var_Importance_Df
# Var_Importance_Df.to_csv("Var_Importance_Df.csv", index = False)

import seaborn as sns
plot = sns.scatterplot(x = "Variable_Name", y = "Value", data = Var_Importance_Df) 
# Task: Rotate the xaxis labels to show up vertically

#################
# RF Model with tuning parameters
#################

M2_RF = RandomForestClassifier(random_state = 123, n_estimators = 25, 
                               max_features = 5, min_samples_leaf = 500)
M2_RF = M2_RF.fit(trainX, trainY)
Test_Pred = M2_RF.predict(testX)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testY, Test_Pred) # R, C format (Actual = testY, Predicted = Test_Pred)
Confusion_Mat 

sum(np.diag(pd.crosstab(testY, Test_Pred)))/testX.shape[0]


#################
# Manual Grid Searching
#################

n_estimators_List = [25, 50, 75] # range(25,100,25)
max_features_List = [5, 7, 9] # range(5,11,2)
min_samples_leaf_List = [100, 200] # range(100,300,100)

Counter = 0

Tree_List = []
Num_Features_List = []
Samples_List = []
Accuracy_List = []

Model_Validation_Df = pd.DataFrame()
Model_Validation_Df2 = pd.DataFrame()
Model_Validation_Df3 = pd.DataFrame()

for i in n_estimators_List:    
    for j in max_features_List:        
        for k in min_samples_leaf_List:                        
            
            Counter = Counter + 1
            print(Counter)
#            print(i,j,k)            
            Temp_Model = RandomForestClassifier(random_state=123, n_estimators = i, 
                                                max_features = j, min_samples_leaf = k)
            Temp_Model = Temp_Model.fit(trainX, trainY)
            Test_Pred = Temp_Model.predict(testX)                 
            Confusion_Mat = pd.crosstab(testY, Test_Pred)
            Temp_Accuracy = (sum(np.diag(Confusion_Mat))/testY.shape[0])*100            
            print(i,i,k,Temp_Accuracy)
            
            # Alternate 1
            Tree_List.append(i)
            Num_Features_List.append(j)
            Samples_List.append(k)
            Accuracy_List.append(Temp_Accuracy)
            
            # Alternate 2
            tempDf = pd.DataFrame([[i,j,k,Temp_Accuracy]]) # [[]] will produce a single row with values, [] will produce single column with values
            Model_Validation_Df2 = Model_Validation_Df2.append(tempDf)
            
            # Alternate 3
            tempDf = pd.DataFrame([[i,j,k,Temp_Accuracy]])
            Model_Validation_Df3 = pd.concat([Model_Validation_Df3, tempDf], axis = 0)
            
            
Model_Validation_Df = pd.DataFrame({'Trees': Tree_List, 'Max_Features': Num_Features_List, 
                                    'Min_Samples': Samples_List, 'Accuracy': Accuracy_List})
    
Model_Validation_Df2.columns = ['Trees', 'Max_Features', 'Min_Samples', 'Accuracy']
Model_Validation_Df3.columns = ['Trees', 'Max_Features', 'Min_Samples', 'Accuracy']


########################################
# Random Forest using GridSearchCV
########################################

from sklearn.model_selection import GridSearchCV

my_param_grid = {'n_estimators': [25, 50, 75], 
                 'max_features': [5, 7, 9], 
                 'min_samples_leaf' : [100, 200]} 

Grid_Search_Model = GridSearchCV(estimator = RandomForestClassifier(random_state=123), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=3).fit(trainX, trainY) # param_grid is a dictionary


Model_Validation_Df4 = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)
# Grid_Search_Model.cv_results_

# Based on the selected hyperparamters, you should build a final model on the COMPLETE training data (trainX, trainY)
RF_Final = RandomForestClassifier(random_state = 123, n_estimators = 50, 
                               max_features = 9, min_samples_leaf = 100).fit(trainX, trainY)
Test_Pred = RF_Final.predict(testX)
