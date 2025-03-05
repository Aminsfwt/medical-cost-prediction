from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the data from the csv file
medical_df = pd.read_csv('E:\ML\Intro to Deep Learning\Labs\Codes\sklearn_tutorial\medical.csv')

#convert smoker column to binary
smoker_codes = {'yes': 1, 'no': 0}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

#convert sex column to binary
sex_codes = {'male': 1, 'female': 0}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)

#create a new column for each region.
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
one_hot = enc.transform(medical_df[['region']]).toarray()
medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

#separate the data depend on smokeing
non_smoker_df = medical_df[medical_df['smoker_code'] == 0]
smoker_df = medical_df[medical_df['smoker_code'] == 1]


#calculate loss function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#model for non smokers 
def non_smoker_model():
    #First, we create a new model object
    non_s_model = LinearRegression()

    #create inputs and target
    non_s_input_data = non_smoker_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']]
    non_s_target_data = non_smoker_df['charges']

    #fit the non_s_model to the data
    non_s_model.fit(non_s_input_data, non_s_target_data)

    """
    #check the model's parameters
    w = model.coef_
    b = model.intercept_
    print('w:', w)
    print('b:', b)
    """
    #predict charges for a group of people (1 sample withe 1 feature)
    non_s_input_examples = non_s_input_data # 2D array(samples, features)
    non_s_predicted_charges = non_s_model.predict(non_s_input_examples)
    #print('Estimated charges for a group of non-smokers :\n', predicted_charges)

    #compute the loss function
    non_smoker_loss = rmse(non_s_predicted_charges, non_s_target_data)
    print('LOSS For Non Smokers: ', non_smoker_loss)

##model for smokers
def smoker_model():
    #First, we create a new model object
    s_model = LinearRegression()

    #create inputs and target
    s_input_data  = smoker_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']]
    s_target_data = smoker_df['charges']

    #fit the s_model to the data
    s_model.fit(s_input_data, s_target_data)

    """
    #check the model's parameters
    w = model.coef_
    b = model.intercept_
    print('w:', w)
    print('b:', b)
    """
    #predict charges for a group of people (1 sample withe 1 feature)
    s_input_examples = s_input_data # 2D array(samples, features)
    s_predicted_charges = s_model.predict(s_input_examples)
    #print('Estimated charges for a group of non-smokers :\n', predicted_charges)

    #compute the loss function
    smoker_loss = rmse(s_predicted_charges, s_target_data)
    print('LOSS For Smokers: ', smoker_loss)

x = non_smoker_model()
y = smoker_model()