import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def datacleaning(df):
    '''
    clean the data for modeling
    Argument:
        df: dataframe
    Returns: cleaned dataframe
    '''
    # Parse Years
    df['year'] = pd.to_datetime(df.issue_d).dt.year
    # Bin late loans into one group
    df.loan_status = df.loan_status.apply(lambda x: np.where(x == 'Late (31-120 days)','Late',x))
    df.loan_status = df.loan_status.apply(lambda x: np.where(x == 'In Grace Period','Late',x))
    df.loan_status = df.loan_status.apply(lambda x: np.where(x == 'Late (16-30 days)','Late',x))
    # Reduce the size of the dataset
    df = df[(df.year.isin([2016,2017,2018]))& 
                        df.loan_status.isin(['Fully Paid','Charged Off'])]
    target_value = {'Fully Paid':1,'Charged Off':0}
    df.loan_status = df.loan_status.replace(target_value)
    df_new = df[[
        'open_acc','total_acc','inq_last_6mths','collections_12_mths_ex_med','tot_coll_amt',
        'revol_bal','pub_rec','pub_rec_bankruptcies','acc_now_delinq','all_util','revol_util','delinq_2yrs',
        'initial_list_status','tot_cur_bal','avg_cur_bal','int_rate','dti','home_ownership','annual_inc','purpose',
        'earliest_cr_line','emp_length','term','addr_state','installment','mort_acc','application_type',
        'verification_status','fico_range_low','fico_range_high','issue_d','year','loan_amnt','grade','loan_status']]
    
    # group some values
    df_new['home_ownership'] = \
            df_new['home_ownership'].apply(lambda x: np.where((x == 'ANY') | (x == 'NONE'), 'OTHER', x))
    df_new['purpose'] = \
            df_new['purpose'].apply(lambda x: np.where((x == 'wedding') | (x == 'renewable_energy'), 'other', x))
    df_new['purpose'] = \
            df_new['purpose'].apply(lambda x: np.where(x == 'house', 'home_improvement', x))
    # remove missing all_util, revol_util, inq_last_6mths, avg_cur_bal
    df_new = df_new[df_new['all_util'].notnull()]
    df_new = df_new[df_new['revol_util'].notnull()]
    df_new = df_new[df_new['inq_last_6mths'].notnull()]
    df_new = df_new[df_new['avg_cur_bal'].notnull()]
    # change value label
    df_new['application_type'] =\
            df_new['application_type'].apply(lambda x: np.where(x == 'Joint App', 'is_joint_app', x))
    # dropping nan and negative values
    df_new['dti'] = df_new['dti'].apply(lambda x: np.where(x<0,np.nan,x))
    df_new = df_new[df_new['dti'].notnull()]
    # number of years since credit line started
    df_new['length_cr_line'] = df_new.year - pd.to_datetime(df_new.earliest_cr_line).dt.year
    # remove rows with missing value 
    df_new = df_new[df_new['emp_length'].notnull()]
    df_new.drop(['year','issue_d','earliest_cr_line'],1,inplace=True)
    
    return df_new

def Dummify(df1, df2):
    '''
    One-Hot encoder dummifies the train and test sets together to resolve the mismatch between them
    Arguments:
        df1 : train set after feature engineering
        df2 : test set after feature engineering
    Returns: Two dummified dataframes, train and test sets
    '''

    # Group the features by dtypes
    train_obj = df1.select_dtypes(["object","category"])
    train_num = df1.select_dtypes(["float64","int64"])
    test_obj = df2.select_dtypes(["object","category"])
    test_num = df2.select_dtypes(["float64","int64"])

    # Apply OneHotEncoder
    encoder = OneHotEncoder(categories = "auto",drop = 'first',sparse = False)
    train_obj_enc = encoder.fit_transform(train_obj)
    test_obj_enc = encoder.transform(test_obj)
    column_name = encoder.get_feature_names(train_obj.columns.tolist())

    # Combine the object and numeric features for train set
    train_df =  pd.DataFrame(train_obj_enc, columns= column_name)
    train_df.set_index(train_num.index, inplace = True)
    train_final = pd.concat([train_df, train_num], axis = 1)

    # Combine the object and numeric features for test set
    test_df =  pd.DataFrame(test_obj_enc, columns= column_name)
    test_df.set_index(test_num.index, inplace = True)
    test_final = pd.concat([test_df, test_num], axis = 1)

    return train_final, test_final

    

    
    
    
    
    