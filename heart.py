from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
import seaborn as sns
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from fcmeans import FCM
from matplotlib import pyplot
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_full = pd.read_csv("heart.csv")
columns = list(df_full.columns)
features = columns[:len(columns) - 1]
# class_labels = list(df_full[columns[-1]])
df1 = df_full[features]
# 
num_attr = len(df1.columns) - 1
# 
k = 2
# Maximum number of iterations
MAX_ITER = 100
# Number of samples
n = len(df1)  # the number of row
# fuzzy parameter
m = 2.00
data=pd.read_csv('heart.csv')
df=pd.read_csv('heart.csv')
print('Data First 5 Rows Show\n')
df.head()
print('Data Last 5 Rows Show\n')
df.tail()
print('Data Show Describe\n')
df.describe()
print('Data Show Info\n')
df.info()
print('Data Show Columns:\n')
df.columns
df.sample(frac=0.01)
#sample; random rows in dataset
df.sample(5)
df=df.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
#New show columns
df.columns
print('Data Shape Show\n')
df.shape
#Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.
print('Data Sum of Null Values \n')
df.isnull().sum()
#all rows control for null values
df.isnull().values.any()

#Data Visualization
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()

sns.pairplot(df)
plt.show()

#Age Analysis
df.Age.value_counts()[:10]
sns.barplot(x=df.Age.value_counts()[:10].index,y=df.Age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.title('Age Analysis System')
plt.show()

#firstly find min and max ages
minAge=min(df.Age)
maxAge=max(df.Age)
meanAge=df.Age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean Age :',meanAge)

young_ages=df[(df.Age>=29)&(df.Age<40)]
middle_ages=df[(df.Age>=40)&(df.Age<55)]
elderly_ages=df[(df.Age>55)]
print('Young Ages :',len(young_ages))
print('Middle Ages :',len(middle_ages))
print('Elderly Ages :',len(elderly_ages))

sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()

#a new feature value can be removed from these age ranges will not affect this impact will see in the future.
df['AgeRange']=0
youngAge_index=df[(df.Age>=29)&(df.Age<40)].index
middleAge_index=df[(df.Age>=40)&(df.Age<55)].index
elderlyAge_index=df[(df.Age>55)].index
for index in elderlyAge_index:
    df.loc[index,'AgeRange']=2
    
for index in middleAge_index:
    df.loc[index,'AgeRange']=1

for index in youngAge_index:
    df.loc[index,'AgeRange']=0

sns.countplot(elderly_ages.Sex)
plt.title("Elderly Sex Operations")
plt.show()

elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum')
sns.barplot(x=elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum').index,y=elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum').values)
plt.title("Gender Group Thalach Show Sum Time")
plt.show()

sns.violinplot(df.Age, palette="Set3", bw=.2, cut=1, linewidth=1)
plt.xticks(rotation=90)
plt.title("Age Rates")
plt.show()

colors = ['blue','green','yellow']
explode = [0,0,0.1]
plt.figure(figsize = (5,5))
plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)],labels=['young ages','middle ages','elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.title('Age States',color = 'blue',fontsize = 15)
plt.show()

df.Sex.value_counts()
sns.countplot(df.Sex)
plt.show()

sns.countplot(df.Sex,hue=df.Slope)
plt.title('Slope & Sex Rates Show')
plt.show()

total_genders_count=len(df.Sex)
male_count=len(df[df['Sex']==1])
female_count=len(df[df['Sex']==0])
print('Total Genders :',total_genders_count)
print('Male Count    :',male_count)
print('Female Count  :',female_count)

#Percentage ratios
print("Male State: {:.2f}%".format((male_count / (total_genders_count)*100)))
print("Female State: {:.2f}%".format((female_count / (total_genders_count)*100)))

#Male State & target 1 & 0
male_andtarget_on=len(df[(df.Sex==1)&(df['Target']==1)])
male_andtarget_off=len(df[(df.Sex==1)&(df['Target']==0)])
####
sns.barplot(x=['Male Target On','Male Target Off'],y=[male_andtarget_on,male_andtarget_off])
plt.xlabel('Male and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

#Female State & target 1 & 0
female_andtarget_on=len(df[(df.Sex==0)&(df['Target']==1)])
female_andtarget_off=len(df[(df.Sex==0)&(df['Target']==0)])
####
sns.barplot(x=['Female Target On','Female Target Off'],y=[female_andtarget_on,female_andtarget_off])
plt.xlabel('Female and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

#Chest Pain Type Analysis
df.Cp.value_counts()  
#0 status at least                           
#1 condition slightly distressed
#2 condition medium problem
#3 condition too bad
sns.countplot(df.Cp)
plt.xlabel('Chest Type')
plt.ylabel('Count')
plt.title('Chest Type vs Count State')
plt.show()

cp_zero_target_zero=len(df[(df.Cp==0)&(df.Target==0)])
cp_zero_target_one=len(df[(df.Cp==0)&(df.Target==1)])
sns.barplot(x=['cp_zero_target_zero','cp_zero_target_one'],y=[cp_zero_target_zero,cp_zero_target_one])
plt.show()

cp_one_target_zero=len(df[(df.Cp==1)&(df.Target==0)])
cp_one_target_one=len(df[(df.Cp==1)&(df.Target==1)])
sns.barplot(x=['cp_one_target_zero','cp_one_target_one'],y=[cp_one_target_zero,cp_one_target_one])
plt.show()

cp_two_target_zero=len(df[(df.Cp==2)&(df.Target==0)])
cp_two_target_one=len(df[(df.Cp==2)&(df.Target==1)])
sns.barplot(x=['cp_two_target_zero','cp_two_target_one'],y=[cp_two_target_zero,cp_two_target_one])
plt.show()

cp_three_target_zero=len(df[(df.Cp==3)&(df.Target==0)])
cp_three_target_one=len(df[(df.Cp==3)&(df.Target==1)])
sns.barplot(x=['cp_three_target_zero','cp_three_target_one'],y=[cp_three_target_zero,cp_three_target_one])
plt.show()

#Target Analysis
#We will analyze this feature for people who are sick or not.
df.Target.unique()
#only two values are shown.
#A value of 1 is the value of patient 0.
sns.countplot(df.Target)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Counter 1 & 0')
plt.show()

sns.countplot(df.Target,hue=df.Sex)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target & Sex Counter 1 & 0')
plt.show()

data_filter_mean=df[(df['Target']==1)&(df['Age']>50)].groupby('Sex')[['Trestbps','Chol','Thalach']].mean()
data_filter_mean.unstack()

X=df.drop('Target',axis=1)
Y=df['Target']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

#principal components analysis
from sklearn.decomposition import PCA
pca=PCA().fit(X_train)
print(pca.explained_variance_ratio_)
print()
print(X_train.columns.values.tolist())
print(pca.components_)
cumulative=np.cumsum(pca.explained_variance_ratio_)
plt.step([i for i in range(len(cumulative))],cumulative)
plt.show()
pca = PCA(n_components=8)
pca.fit(X_train)
reduced_data_train = pca.transform(X_train)
#inverse_data = pca.inverse_transform(reduced_data)
plt.scatter(reduced_data_train[:, 0], reduced_data_train[:, 1], label='reduced')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
pca = PCA(n_components=8)
pca.fit(X_test)
reduced_data_test = pca.transform(X_test)
#inverse_data = pca.inverse_transform(reduced_data)
plt.scatter(reduced_data_test[:, 0], reduced_data_test[:, 1], label='reduced')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

reduced_data_train = pd.DataFrame(reduced_data_train, columns=['Dim1', 'Dim2','Dim3','Dim4','Dim5','Dim6','Dim7','Dim8'])
reduced_data_test = pd.DataFrame(reduced_data_test, columns=['Dim1', 'Dim2','Dim3','Dim4','Dim5','Dim6','Dim7','Dim8'])
X_train=reduced_data_train
X_test=reduced_data_test

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# Initialize the fuzzy matrix U
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]  #First normalization
        membership_mat.append(temp_list)
    return membership_mat


# 
def calculateClusterCenter(membership_mat):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(k):
        x = cluster_mem_val_list[j]
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]  # Each dimension must be calculated.
        cluster_centers.append(center)
    return cluster_centers


#Update membership
def updateMembershipValue(membership_mat, cluster_centers):
    #    p = float(2/(m-1))
    data = []
    for i in range(n):
        x = list(df.iloc[i])  # Take out each line of data in the file
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j] / distances[c]), 2) for c in range(k)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat, data


# Get cluster results
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # 
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:  # The maximum number of iterations
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat, data = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    print(membership_mat)
    return cluster_labels, cluster_centers, data, membership_mat


def xie_beni(membership_mat, center, data):
    sum_cluster_distance = 0
    min_cluster_center_distance = inf
    for i in range(k):
        for j in range(n):
            sum_cluster_distance = sum_cluster_distance + membership_mat[j][i] ** 2 * sum(
                power(data[j, :] - center[i, :], 2))  # 
    for i in range(k - 1):
        for j in range(i + 1, k):
            cluster_center_distance = sum(power(center[i, :] - center[j, :], 2))  # Calculate the distance between classes
            if cluster_center_distance < min_cluster_center_distance:
                min_cluster_center_distance = cluster_center_distance
    return sum_cluster_distance / (n * min_cluster_center_distance)


labels, centers, data, membership = fuzzyCMeansClustering()
print(labels)
print(centers)
center_array = np.array(centers)
label = np.array(labels)
datas = np.array(data)

print("Cluster validity:", xie_beni(membership, center_array, datas))

model = FCM(n_clusters=2)
model.fit(X)
yhat = model.predict(X)

clusters = unique(yhat)

for cluster in clusters:
	
	row_ix = where(yhat == cluster)
	
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
pyplot.title('Fuzzy C Means Cluster')
pyplot.xlabel('Actual')
pyplot.ylabel('predicted')
pyplot.show()