#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame
from sklearn import preprocessing 
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
import csv
from csv import writer
from csv import reader


# In[ ]:





# In[155]:


col_names = [ 'player_name', 'team_abbreviation',
             'age', 'player_height', 'player_weight',
             'college', 'country', 'draft_year', 'draft_round', 
             'draft_number', 'gp', 'pts', 'reb', 'ast', 'net_rating', 
             'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct', 
             'season', 'data_pts', 'data_age', 'data_weight','data_height',
             'data_gp']
seasons = pd.read_csv('all_seasons.csv', names = col_names)


# In[156]:


seasons.head()


# In[157]:


#player points
with open('all_seasons.csv', 'r') as read_obj: 
    next(read_obj)
    with open ('new_seasons.csv', 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        
        for row in csv_reader:
            if float(row[12]) > 0.0 and float(row[12]) < 10.0:
                row.append("1")
            if float(row[12]) > 10.0 and float(row[12]) < 20.0:
                row.append("2") 
            if float(row[12]) > 20.0 and float(row[12]) < 30.0:
                row.append('3') 
            if float(row[12]) > 30.0 and float(row[12]) < 40.0:
                row.append('4') 
            #if float(row[12]) > 25.0 and float(row[12]) < 30.0:
                #row.append('5')  
            #if float(row[12]) > 30.0 and float(row[12]) < 40.0:
                #row.append('6')  
            csv_writer.writerow(row)
            
            
new_pts = pd.read_csv('new_seasons.csv', header=None, names=col_names)
new_pts.head()
#print(new_pts)


# In[158]:


print(new_pts['data_pts'].describe())
#Every 6th points


# In[159]:


#player_age
with open('new_seasons.csv', 'r') as read_obj: 
    next(read_obj)
    with open ('new_seasons1.csv', 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        
        for row in csv_reader:
            if float(row[3]) > 18.0 and float(row[3]) < 23.0:
                row.append("1")
            if float(row[3]) > 24.0 and float(row[3]) < 29.0:
                row.append("2") 
            if float(row[3]) > 30.0 and float(row[3]) < 35.0:
                row.append('3') 
            if float(row[3]) > 36.0 and float(row[3]) < 41.0:
                row.append('4') 
            if float(row[3]) > 42.0 and float(row[3]) < 47.0:
                row.append('5')  
            csv_writer.writerow(row)
new_pts1 = pd.read_csv('new_seasons1.csv', header=None, names=col_names)
new_pts1.head()            


# In[160]:


print(new_pts1['data_age'].describe())


# In[161]:


#player_weight
with open('new_seasons1.csv', 'r') as read_obj: 
    next(read_obj)
    with open ('new_seasons2.csv', 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        for row in csv_reader:
            if float(row[5]) > 60.0 and float(row[5]) < 80.0:
                row.append("1")
            if float(row[5]) > 70.0 and float(row[5]) < 110.0:
                row.append("2")
            if float(row[5]) > 110.0 and float(row[5]) < 130.0:
                row.append("2")
            if float(row[5]) > 130.0 and float(row[5]) < 150.0:
                row.append('3')
            if float(row[5]) > 150.0 and float(row[5]) < 164.0:
                row.append('4')
            csv_writer.writerow(row)
new_pts2 = pd.read_csv('new_seasons2.csv', header=None, names=col_names)
new_pts2.head()


# In[162]:
print(row[4])




# In[163]:


#player_height
with open('new_seasons2.csv', 'r') as read_obj: 
    next(read_obj)
    with open ('new_seasons3.csv', 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        for row in csv_reader:
            if float(row[4]) > 160.0 and float(row[4]) < 189.0:
                row.append("1")
            if float(row[4]) > 190.0 and float(row[4]) < 200.0:
                row.append("2") 
            if float(row[4]) > 201.0 and float(row[4]) < 208.0:
                row.append('3') 
            if float(row[4]) > 209.0 and float(row[4]) < 217.0:
                row.append('4') 
            if float(row[4]) > 217.0 and float(row[4]) < 232.0:
                row.append('5')  
            csv_writer.writerow(row)
new_pts3 = pd.read_csv('new_seasons3.csv', header=None, names=col_names)
new_pts3.head()


# In[164]:
#games played
with open('new_seasons3.csv', 'r') as read_obj: 
    next(read_obj)
    with open ('new_seasons4.csv', 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        for row in csv_reader:
            if float(row[11]) > 0.0 and float(row[11]) < 40.0:
                row.append("1")
            if float(row[11]) > 40.0 and float(row[11]) < 60.0:
                row.append("2")
            if float(row[11]) > 60.0 and float(row[11]) < 90.0:
                row.append('3')
            csv_writer.writerow(row)
new_pts4 = pd.read_csv('new_seasons4.csv', header=None, names=col_names)
new_pts4.head()




new_pts4['age'].fillna(new_pts4['age'].mode()[0], inplace = True)
new_pts4['player_height'].fillna(new_pts4['player_height'].mode()[0], inplace = True)
new_pts4['player_weight'].fillna(new_pts4['player_weight'].mode()[0], inplace = True)
new_pts4['draft_year'].fillna(new_pts4['draft_year'].mode()[0], inplace = True)
new_pts4['draft_round'].fillna(new_pts4['season'].mode()[0], inplace = True)
new_pts4['season'].fillna(new_pts4['season'].mode()[0], inplace = True)
new_pts4['data_pts'].fillna(new_pts4['data_pts'].mode()[0], inplace = True)
new_pts4['data_age'].fillna(new_pts4['data_age'].mode()[0], inplace = True)
new_pts4['data_weight'].fillna(new_pts4['data_weight'].mode()[0], inplace = True)
new_pts4['data_height'].fillna(new_pts4['data_height'].mode()[0], inplace = True)
new_pts4['data_gp'].fillna(new_pts4['data_gp'].mode()[0], inplace = True)
new_pts4.isnull().sum()


# In[ ]:





# In[266]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
new_pts4['season'] = lb_make.fit_transform(new_pts4['season'])
new_pts4['draft_round'] = lb_make.fit_transform(new_pts4['draft_round'])
new_pts4['draft_year'] = lb_make.fit_transform(new_pts4['draft_year'])
new_pts4['team_abbreviation'] = lb_make.fit_transform(new_pts4['team_abbreviation'])


# In[267]:


#new_pts = seasons.drop(seasons.index[0])
new_pts4.head()


# In[167]:


#seasons[['data']].describe()


# In[295]:


#removing season and team abbreviation draft year and draft round
features = ['age','player_height','team_abbreviation','draft_year','season']
x = new_pts4.loc[:, features]
y = new_pts4.loc[:,['data_pts', 'data_gp']]


# In[296]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# In[297]:


print(type(X_train), X_train.size, X_train.shape)
print(type(Y_train), Y_train.size, Y_train.shape)
print(type(X_test), X_test.size, X_test.shape)
print(type(Y_test), Y_test.size, Y_test.shape)


# In[298]:


tf.convert_to_tensor(Y_train, np.float)
tf.convert_to_tensor(X_train, np.float)


# In[299]:


print(type(X_train), X_train.size, X_train.shape)
print(type(Y_train), Y_train.size, Y_train.shape)
print(type(X_test), X_test.size, X_test.shape)
print(type(Y_test), Y_test.size, Y_test.shape)


# In[306]:


X_train=np.asarray(X_train).astype(np.float32)
X_test=np.asarray(X_test).astype(np.float32)
Y_train=np.asarray(Y_train).astype(np.float32)
Y_test=np.asarray(Y_test).astype(np.float32)


# In[307]:


print(type(X_train), X_train.size, X_train.shape)
print(type(Y_train), Y_train.size, Y_train.shape)
print(type(X_test), X_test.size, X_test.shape)
print(type(Y_test), Y_test.size, Y_test.shape)


# In[308]:


from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()
model.add(Dense(27, input_dim=5, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# In[309]:


model.compile(loss='binary_crossentropy', 
              optimizer='nadam', metrics=['accuracy'])


# In[310]:


model.summary()


# In[311]:


model.fit(X_train, Y_train, epochs=100, batch_size =30)


# In[264]:


np.argmax(model.predict(x), axis=1)


# In[265]:


model.evaluate(X_test, Y_test)[1]


# In[ ]:




