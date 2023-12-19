import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df['landmass'].value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df['landmass'].isin([3,6])]

#Print the average vales of the predictors for Europe and Oceania
print(df_36.groupby('landmass')[var].mean().T)

#Create labels for only Europe and Oceania
df_36 = df[df["landmass"].isin([3,6])]
labels = df_36["landmass"]

#Print the variable types for the predictors
print(df_36[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df_36[var])

#Split data into a train and test set

x_train,x_test,y_train,y_test=train_test_split(data,labels,random_state=1,test_size=0.4)
#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []
for i in range(1,21):
  model=DecisionTreeClassifier(random_state=1,max_depth=i)
  model.fit(x_train,y_train)
  score=model.score(x_test,y_test)
  acc_depth.append(score)
print(acc_depth)
#Plot the accuracy vs depth
plt.plot(range(1,21), acc_depth)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()

idx=acc_depth.index(max(acc_depth))

#Find the largest accuracy and the depth this occurs
#Refit decision tree model with the highest accuracy and plot the decision tree
model=DecisionTreeClassifier(random_state=1,max_depth=idx+1)
model.fit(x_train,y_train)
tree.plot_tree(model, feature_names = x_train.columns,  
               class_names = ['Europe', 'Oceania'],
                filled=True)
plt.show()
#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
acc_pruned = []
ccp = np.logspace(-3, 0, num=20)
for i in ccp:
    dt_prune = DecisionTreeClassifier(random_state = 1, max_depth = idx+1, ccp_alpha=i)
    dt_prune.fit(x_train, y_train)
    acc_pruned.append(dt_prune.score(x_test, y_test))

#Plot the accuracy vs ccp_alpha
plt.plot(ccp, acc_pruned)
plt.xscale('log')
plt.xlabel('ccp_alpha')
plt.ylabel('accuracy')
plt.show()


#Find the largest accuracy and the ccp value this occurs
max_acc_pruned = np.max(acc_pruned)
best_ccp = ccp[np.argmax(acc_pruned)]

print(f'Highest accuracy {round(max_acc_pruned,3)*100}% at ccp_alpha {round(best_ccp,4)}')

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dt_final = DecisionTreeClassifier(random_state = 1, max_depth = idx+1, ccp_alpha=best_ccp)
dt_final.fit(x_train, y_train)
