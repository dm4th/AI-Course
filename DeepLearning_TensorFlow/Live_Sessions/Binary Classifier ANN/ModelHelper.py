#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error
import joblib

def data_split_standardise(x,y=None):
    if y is None:
        st=StandardScaler()
        st.fit(x)
        x_std=st.transform(x)
        joblib.dump(st,"StandardScalar_trained.h5")
        return(x_std)
    else:

        
        
        
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
        st=StandardScaler()
        st.fit(x_train)
        x_train_std=st.transform(x_train)
        x_test_std=st.transform(x_test)
        joblib.dump(st,"StandardScalar_trained.h5")    
        return(x_train_std,x_test_std,y_train,y_test)


# In[37]:


def model_performance_classification(model,train,test):
    x_train,y_train=train
    x_test,y_test = test
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("############### Classification Report: Train Data #############################")
    print(classification_report(y_train,y_train_pred))
    print("                                                                               ")
    print("############### Classification Report: Test Data #############################")
    print(classification_report(y_test,y_test_pred))
    


# In[2]:


def model_performance_regression(model,train,test):
    x_train,y_train=train
    x_test,y_test = test
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    print("############### R- Squared #############################")
    print("Train: ",model.score(y_train,y_train_pred))
    print("Test: ",model.score(y_test,y_test_pred))
    print("                                                                               ")

    print("############### Adjusted R- Squared #############################")
    r2_train = model.score(y_train,y_train_pred)
    ad_r2_train = 1 - ((1-r2_train)*(x_train.shape[0]-1)/(x_train.shape[0]-x_train.shape[1]-1))
    r2_test = model.score(y_test,y_test_pred)
    ad_r2_test = 1 - ((1-r2_test)*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))
    print("Train: ",ad_r2_train)
    print("Test: ",ad_r2_test)
    print("                                                                               ")

    print("############### Mean Absolute Error #############################")
    print("Train: ",mean_absolute_error(y_train,y_train_pred))
    print("Test: ",mean_absolute_error(y_test,y_test_pred))
    print("                                                                               ")
    print("############### Mean Absolute Error #############################")
    print("Train: ",mean_squared_error(y_train,y_train_pred))
    print("Test: ",mean_squared_error(y_test,y_test_pred))
    print("                                                                               ")
    
    


# In[ ]:




