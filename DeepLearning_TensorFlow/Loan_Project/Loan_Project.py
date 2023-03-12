#!/usr/bin/env python
# coding: utf-8

# # Lending Club Loan Project
# ### Danny Mathieson - March 2022

# ### Downloading and Displaying the Dataset

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('datasets/loan_data.csv')
df.head()


# In[4]:


df.shape


# In[5]:


# check for nulls
df.isna().sum()


# ### 1. Feature Transformation - Transform Categorical Values into Numerical Values

# In[6]:


df.dtypes


# In[7]:


# Only purpose needs to be changed to numerical values - get dummies
df_dummy = pd.get_dummies(df, drop_first=True)
df_dummy.head()


# In[8]:


df_dummy.columns


# ### 2. EDA on Different Factors of the Dataset

# In[9]:


# Describe the data, split columns into either binary or numerical sub-types
df_dummy.describe()
numerical_cols = ['int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util','inq.last.6mths', 'delinq.2yrs', 'pub.rec']
binary_cols = ['credit.policy','purpose_credit_card', 'purpose_debt_consolidation','purpose_educational','purpose_home_improvement','purpose_major_purchase','purpose_small_business']


# In[10]:


from matplotlib import pyplot as plt
import seaborn as sns
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[11]:


# Loop over numerical columns - Box Plot overall and by fully paid status
for c in numerical_cols:
    # get datasets by category
    total_data = df_dummy[[c]]
    cat_data = df_dummy[[c, 'not.fully.paid']]
    # create boxplots
    sns.boxplot(x=None, y=c, data=total_data, ax=plt.subplot(1,2,1))
    sns.boxplot(x='not.fully.paid', y=c, data=cat_data, ax=plt.subplot(1,2,2))
    # format chart and show
    plt.suptitle(c)
    plt.xlabel('Not Paid Status')
    plt.show()


# ##### - int.rate
#     - Generally int.rate seems to correspond with a higher liklihood of not paying as the 25%, Median, and 75% are all higher than the paid class by a full percent or two
#     - The upper extreme doesn't seem to impact the outcome much as the max percent excluding outliers is the same and they seem to have even numbers of outliers on the high end
#     - interesting how no one with a sub 7% rate defaulted  
# ##### - installment
#     - Once we start to get above 3.5 years the liklihood of default seems to go up, especially above 5 years  
# ##### - log.annual.inc - nothing  
# ##### - dti - nothing  
# ##### - fico
#     - On first glance, lower fico scores definitely seem to have an impact on payback liklihood, but the entire range of paid back loans' fico scores contains the range of not paid back loans, including outliers.  
# ##### - days.with.cr.line - nothing  
# ##### - revol.bal - scale is too messed up to see much  
# ##### - revol.util
#     - higher utilization rates has a slight impact on not paying back fully  
# ##### - inq.last.6mths
#     - slightly impactful if above 2  
# ##### - delinq.2yrs 
#     - not enough non-zero data
#     - more outliers that have paid pack than haven't  
# #####  pub.rec 
#     - not enough non-zero data
#     - more outliers that have paid pack than haven't  

# In[12]:


# Loop over Binary Columns & create a bar plot & stacked bar plot 
for c in binary_cols:
    bin_data = df_dummy[[c, 'not.fully.paid']]
    bin_data['Paid Status'] = np.where(df_dummy['not.fully.paid'] == 0, 'No', 'Yes')
    bin_data['Condition'] = np.where(df_dummy[c] == 0, 'No', 'Yes')
    # calculate percentages
    percent = bin_data.groupby(['Condition', 'Paid Status']).size().reset_index(name='count')
    percent['pct'] = percent['count'] / percent['count'].sum() * 100
    # create the plot
    order = {
        'Paid Status': ['No','Yes'],
        'Condition': ['No','Yes']
    }
    axis = sns.countplot(x='Paid Status', hue='Condition', data=bin_data, order=order['Paid Status'], hue_order=order['Condition'])
    # add percentages for tooltips
    counter=0
    for p in axis.patches:
        h = p.get_height()
        pct = f"{round(percent['pct'][counter],1)}%"
        x_ax_pos = p.get_x() + p.get_width() / 2.0
        counter += 1
        axis.text(
            x_ax_pos,
            h + 3,
            pct,
            ha='center'
        )
    plt.title(c)
    plt.xlabel('Not Paid Status')
    plt.ylabel('Loans')
    plt.legend()
    plt.show()
    


# ##### - credit.policy
#     - About 4/5 of all peope are approved for credit
#     - About 2/3 of the unpaid loans are from people that fit the credit policy criteria
#     - In general unapproved people are more likely to default. 25% of unapproved loans defaulted compared to just 13% of approved loans
# ##### - purpose_credit_card
#     - 13.2% of loans are credit card loans
#     - 11.4% of credit card loans default
# ##### - purpose_debt_consolidation 
#     - 41.3% of loans are debt consolidation
#     - 15.2% of debt consolidation loans default
# ##### - purpose_educational 
#     - Only 3.6% of loans are educational
#     - 19.4% of educational loans default
# ##### - purpose_home_improvement
#     - 6.5% of loans are home improvement
#     - 16.9% of home improvement loans default
# ##### - purpose_major_purchase
#     - 4.6% of loans are for a major purchase
#     - 10.9% of major purchase loans default
# ##### - purpose_small_business
#     - 6.5% of loans are for a small business
#     - 27.7% of small business loans default
#     
# #### - Key Takeaways
#     - Small Business loans are by far the riskiest, followed by educational
#     - Debt Consolidation is 41.3% of all loans, and they default at a slightly lower rate than average (16% default) 
#     - Educational makes up such a small percentage, that small business is likely not super impactful
#     - The only other loan type that is above average is Home Improvement

# ### 3. Additional Feature Engineering

# In[13]:


# Find correlations between numerical features
features = df_dummy.drop(columns=['not.fully.paid'], axis=1)
num_features = features[numerical_cols]
labels = df_dummy[['not.fully.paid']]


# In[14]:


num_corr = num_features.corr()
sns.heatmap(data=num_corr, square=True, cmap='bwr')


# In[15]:


corr_arr = num_corr.unstack()
corr_arr = corr_arr[corr_arr != 1]
corr_arr = corr_arr.drop_duplicates()
sorted_corr = corr_arr.sort_values(ascending=False)
opp_sorted_corr = corr_arr.sort_values(ascending=True)
print(f'Top Positive Correlations:\n\n{sorted_corr.head(10)}')
print(f'\n\nTop Negative Correlations:\n\n{opp_sorted_corr.head(10)}')


# #### Correlation takeaways:
# - The positive correlations are all below 0.5. I think we shouldn't remove any features due to those correlations
# - FICO score is negatively correlated with quite a few features and should be removed because of it
# - I may come back to check on int.rate & revol.util after building the model

# In[16]:


features = features.drop(columns=['fico'], axis=1)


# ### 4. Modeling

# In[17]:


# Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve,confusion_matrix
import joblib

def data_split_standardise(x,y=None):
    if y is None:
        st=StandardScaler()
        st.fit(x)
        x_std=st.transform(x)
        joblib.dump(st,"model_objects/StandardScalar_trained.h5")
        return(x_std)
    else:
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
        st=StandardScaler()
        st.fit(x_train)
        x_train_std=st.transform(x_train)
        x_test_std=st.transform(x_test)
        joblib.dump(st,"model_objects/StandardScalar_trained.h5")    
        return(x_train_std,x_test_std,y_train,y_test)


# In[18]:


x_train, x_test, y_train, y_test = data_split_standardise(features,labels)


# In[19]:


# Build Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, AUC
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from livelossplot import PlotLossesKerasTF


# In[20]:


# Modeling Constants
METRICS = [
    BinaryAccuracy(name='Binary_Accuracy'),
    Precision(name='Precision'),
    Recall(name='Recall'),
    TruePositives(name='True_Positives'),
    TrueNegatives(name='True_Negatives'),
    FalsePositives(name='False_Positives'),
    FalseNegatives(name='False_Negatives'),
    AUC(name='AUC'),
    AUC(name='Precision-Recall', curve='PR')
]
EPOCHS = 100
BATCH_SIZE = 512


# In[21]:


# Helpful plotting functions

# Confusion Matrix
def plot_cm(y_act, y_pred, p=0.5):
    cm = confusion_matrix(y_act, y_pred > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.title(f'Confusion Matrix P={p}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

# ROC Curve
def plot_roc(y_act, y_pred, name='ROC', **kwargs):
    false_positive, true_positive, na = roc_curve(y_act, y_pred)

    plt.plot(false_positive, true_positive, label=name, **kwargs)
    plt.xlabel('False Positives')
    plt.ylabel('True Positives')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

# Precision-Recall Curve
def plot_prc(y_act, y_pred, name='ROC', **kwargs):
    precision, recall, _ = precision_recall_curve(y_act, y_pred)

    plt.plot(precision, recall, label=name, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


# In[22]:


# First Attempt
def make_basic_model(metrics=METRICS, output_bias=None):
    if output_bias:
        output_bias = Constant(output_bias)

    model = Sequential()
    model.add(Input(shape=(None,x_train.shape[1]),name='Input_Layer'))
    model.add(Dense(12,activation='relu',name='Hidden_Layer_1'))
    model.add(Dense(8,activation='relu',name='Hidden_Layer_2'))
    model.add(Dense(1,activation='sigmoid',name='Output_Layer',bias_initializer=output_bias))

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.001),
        metrics=METRICS
    )

    return model


# In[23]:


model = make_basic_model(metrics=METRICS)


# In[24]:


model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test,y_test),
    callbacks=[PlotLossesKerasTF()]
)


# In[25]:


# Helper function for fitting & evaluating models
def evaluate_and_plot(model, x_train, x_test, y_train, y_test, batch_size=BATCH_SIZE):
    # Get Predictions & Evaluate
    train_preds = model.predict(x_train, batch_size=batch_size)
    test_preds = model.predict(x_test, batch_size=batch_size)
    results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    # Print Metric Scores
    print('\n\n')
    for metric, value in zip(model.metrics_names, results):
        print(f'{metric}:\t{value}')

    # Plot Confusion Matrix
    print('\n\n')
    plot_cm(y_test, test_preds)
    plt.show()

    # Plot ROC
    print('\n\n')
    plot_roc(y_train, train_preds, name='Train', color=colors[0])
    plot_roc(y_test, test_preds, name='Test', color=colors[1])
    plt.legend()
    plt.show()

    # Plot Precision-Recall
    print('\n\n')
    plot_prc(y_train, train_preds, name='Train', color=colors[0])
    plot_prc(y_test, test_preds, name='Test', color=colors[1])
    plt.legend()
    plt.show()


# In[26]:


evaluate_and_plot(
    model,
    x_train,
    x_test,
    y_train,
    y_test
)


# ### Model Overfit
# Accuracy was great because we didn't predict any defaults - Class Imbalance
# 
# Let's try to add some initial bias

# In[27]:


# set initial bias
neg, pos = np.bincount(labels['not.fully.paid'])
initial_bias = np.log([pos/neg])
initial_bias


# In[28]:


model = make_basic_model(metrics=METRICS, output_bias=initial_bias)

model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test,y_test),
    callbacks=[PlotLossesKerasTF()]
)


# In[29]:


evaluate_and_plot(
    model,
    x_train,
    x_test,
    y_train,
    y_test
)


# #### Model Still Overfit - We actually Got Some Non-Payers Though!
# Let's add some class weights to try to make sure we catch more fraud

# In[30]:


neg_weight = (1 / neg) * (labels.shape[0] / 2.0)
pos_weight = (1 / pos) * (labels.shape[0] / 2.0)
weights = {0: neg_weight, 1: pos_weight}
weights


# In[31]:


model = make_basic_model(metrics=METRICS, output_bias=initial_bias)

model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test,y_test),
    callbacks=[PlotLossesKerasTF()],
    class_weight=weights
)


# In[32]:


evaluate_and_plot(
    model,
    x_train,
    x_test,
    y_train,
    y_test
)


# Loss improved a bit, but still definitely overfit and in need of a bit more work

# #### Additional Feature Engineering
# 
# Let's remove ```log.annual.inc``` and ```revol.util``` and look at correlations again

# In[33]:


non_feature_cols = ['not.fully.paid','fico','log.annual.inc','revol.util']
features = df_dummy.drop(columns=non_feature_cols, axis=1)
num_features = features.drop(columns=binary_cols, axis=1)


# In[34]:


num_corr = num_features.corr()
sns.heatmap(data=num_corr, square=True, cmap='bwr')


# In[35]:


corr_arr = num_corr.unstack()
corr_arr = corr_arr[corr_arr != 1]
corr_arr = corr_arr.drop_duplicates()
sorted_corr = corr_arr.sort_values(ascending=False)
opp_sorted_corr = corr_arr.sort_values(ascending=True)
print(f'Top Positive Correlations:\n\n{sorted_corr.head(10)}')
print(f'\n\nTop Negative Correlations:\n\n{opp_sorted_corr.head(10)}')


# In[36]:


# Let's model with the same structure as the last run we did
x_train, x_test, y_train, y_test = data_split_standardise(features,labels)


# In[37]:


model = make_basic_model(metrics=METRICS, output_bias=initial_bias)

model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test,y_test),
    callbacks=[PlotLossesKerasTF()],
    class_weight=weights
)


# In[38]:


evaluate_and_plot(
    model,
    x_train,
    x_test,
    y_train,
    y_test
)


# Still not quite what we'd like... Seems like we're now predicting too many fraud cases.
# 
# ### Oversampling
# 
# Try oversampling on the positive class to make sure we're identifying potential loans that won't be repaid

# In[58]:


# Set up split datasets between pos and neg observations to sample at different rates from them
pos_df = df_dummy[df_dummy['not.fully.paid'] == 1].reset_index()
neg_df = df_dummy[df_dummy['not.fully.paid'] == 0].reset_index()

# pos_features = pos_df.drop(columns=non_feature_cols, axis=1)
# neg_features = neg_df.drop(columns=non_feature_cols, axis=1)

# pos_labels = pos_df['not.fully.paid']
# neg_labels = neg_df['not.fully.paid']


# In[61]:


pos_df.index


# In[63]:


# Randomly Sample the same number of positive observations as we have in the negative set
resampled_pos_df = pos_df.sample(
    n=neg_df.shape[0],
    replace=True,
    random_state=0
)
resampled_pos_df.shape


# In[68]:


# re-combine the newly resampled dataset
resampled_df = pd.concat([resampled_pos_df, neg_df])

resampled_features = resampled_df.drop(columns=non_feature_cols, axis=1)
resampled_labels = resampled_df['not.fully.paid']

resampled_df.shape


# In[69]:


# confirm re-balanced classes
resampled_df['not.fully.paid'].value_counts()


# In[70]:


# split the data into train & test
x_train, x_test, y_train, y_test = data_split_standardise(resampled_features,resampled_labels)


# In[72]:


# re-build the model
# make sure to not add class weights or initial bias because we've rebalanced already
model = make_basic_model(metrics=METRICS, output_bias=initial_bias)

model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test,y_test),
    callbacks=[PlotLossesKerasTF()]
)


# In[73]:


evaluate_and_plot(
    model,
    x_train,
    x_test,
    y_train,
    y_test
)


# Much better results!!

# 
