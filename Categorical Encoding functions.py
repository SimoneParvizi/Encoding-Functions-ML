import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn import base
from sklearn.model_selection import KFold


# The importance of this encoding is briefly explain in the following link https://www.youtube.com/watch?v=589nCGeWG1w&ab_channel=StatQuestwithJoshStarmer
# The weighted mean transform the cat in the column in a continuous value



# The next function does K-Fold Target Encoding column for column.
# For the implementation of K-Fold Target Encoding for the whole dataset, see the function below


def Kfold_target_encoding(df, column_to_encode, target, weight_m, n_fold):

    encoded_dataset = df.copy()

    kfold = KFold(n_splits = n_fold) # Default shuffle = False, random_state=None
    list_of_indexes = [val_ind for I, (tr_ind, val_ind) in enumerate(kfold.split(df))]

    last_index = (len(list_of_indexes) - 1)    
    unique_values = df[column_to_encode].unique()
    
    for number_folds in range(n_fold):
    
        first_fold = df.iloc[list_of_indexes[0]]
        current_fold = df.iloc[list_of_indexes[number_folds]]

    
        # If it's the last fold, then the next fold is the first fold 
        if number_folds == last_index:
            next_fold = first_fold
        # Else the next fold is the next fold
        else:  
            next_fold = df.iloc[list_of_indexes[number_folds + 1]]    
            
        
        for each_category in unique_values:
            if each_category not in next_fold[column_to_encode].unique(): 
                over_all_mean = next_fold[target].mean() 
                weighted_mean = (0 + weight_m * over_all_mean) / (0 + weight_m)          


            else: 
                counts = next_fold.groupby(column_to_encode)[target].count()
                mean_target_feature = next_fold.groupby(column_to_encode)[target].mean()
                over_all_mean = next_fold[target].mean()
                weighted_mean = (counts[each_category] * mean_target_feature[each_category] + weight_m * over_all_mean) / (counts[each_category] + weight_m)
            

            
            # Replacing the cat values of the fold with the weighted mean
            current_fold.loc[current_fold[column_to_encode] == each_category, column_to_encode] =  weighted_mean
            

        
        # Append only rows with float values in "color" column to encoded_dataset dataframe and save it as encoded_dataset  
        encoded_dataset = encoded_dataset.append(current_fold[current_fold[column_to_encode].apply(lambda x: isinstance(x, float))])  
        
        # Remove from encoded_dataset dataframe the rows with non-float values in "color" column
        encoded_dataset = encoded_dataset[encoded_dataset[column_to_encode].apply(lambda x: isinstance(x, float))]    

    return encoded_dataset











# K-Fold Target Encoding for the whole dataset  

def Kfold_target_encoding_all_columns(df, target, weight_m, n_fold):

        # Be sure that the target is encoded as 0 and 1 (Ordinal Encoder)
        # Use the attribute 'mapping' to be sure that is mapped as 0 and 1
        encoder = ce.OrdinalEncoder(cols=[target])
        df = encoder.fit_transform(df)

        all_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        encoded_dataset = df.copy()
        
        for each_column in all_categorical_columns:
            encoded_dataset = Kfold_target_encoding(encoded_dataset, each_column, target, weight_m, n_fold)
            
        return encoded_dataset






# This function is check whether the TargetEncode gives the same results of what the 
# Smooth encoding should give
 

def Smooth_encoding(df, column_to_encode, target, weight):
    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(column_to_encode)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    return df[column_to_encode].map(smooth)




def Bayesian_target_encoder(df, target, smooth_value):

    all_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Be sure that the target is encoded as 0 and 1 (Ordinal Encoder)
    # Use the attribute 'mapping' to be sure that is mapped as 0 and 1
    encoder = ce.OrdinalEncoder(cols=[target])
    df = encoder.fit_transform(df)

    encoder = ce.TargetEncoder(cols=all_categorical_columns, smoothing=smooth_value)
    df = encoder.fit_transform(df, df[target])

    return df