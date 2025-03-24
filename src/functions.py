
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline


def clean_data(data_df: pd.DataFrame):
    """
    we use this function to clean our raw data.
    More info on what we keep can be found on the notebook
    """

    # we check for NAs
    data_df.fillna(value=0.0)

    # we remove the duplicates
    data_df=remove_duplicates(data_df)

    # we keep the BMI,Age and sex
    bmi=data_df["BMI"]
    age=data_df["Host age"]
    sex=data_df["Sex"]
    
    # we keep the bacteria only
    to_drop=['Unnamed: 0', 'Project ID','Experiment type','Sex','Host age','Disease MESH ID','BMI']
    bacteria=data_df.drop(labels=to_drop,axis=1)

    # we encode the sex
    sex_encoded=sex.apply(encode_sex)

    # we get the bacteria names
    bacteria_names=list(bacteria)

    # we scale the bacteria data
    scaled_bacteria=scale_data(bacteria,bacteria_names)

    frames=[bmi,age,sex_encoded,scaled_bacteria]

    return frames



def remove_duplicates(data_df: pd.DataFrame):
    """
    We use this function to any potential duplicates
    """
    
    df=data_df
   
    shape_before=df.shape
    df.drop_duplicates()
    shape_after=df.shape

    if (shape_before[0] != shape_after[0]):
        print("Before removal of duplicates",shape_before)
        print("After removal of duplicates",shape_after)
    else:
        print("No duplicates in the set")
    
    return df

def encode_sex(sex):
    """
    Method used to encode the entries of the column sex
    Male --> 0
    Female --> 1
    Other --> 2
    """

    if sex=="Male":
        return 0
    elif sex=="Female":
        return 1
    else:
        return 2

def scale_data(data_df: pd.DataFrame,names):
    """
    We use this function to scale our data.
    Further commenting can be found on the notebook
    """

    pipeline = Pipeline([
    ('scaler', RobustScaler()),  
    ('transformer', PowerTransformer(method='yeo-johnson'))
    ])

    scaled_data=pipeline.fit_transform(data_df) # type: ignore
    scaled_data_df=pd.DataFrame(columns=names,data=scaled_data)
    scaled_data_df=scaled_data_df.reindex(sorted(scaled_data_df.columns),axis=1)
    
    return scaled_data_df


def save_clean_data(frames,mode,data_path):
    """
    We use this function to save our newly cleaned data
    """

    file_name=data_path
    if mode=="Eval":
        file_name += "/evaluation_final_data.csv"
    else:
        file_name += "/development_final_data.csv"
    
    print("Creating dataframe for the clean data...")
    to_be_saved=pd.concat(frames,axis=1)
    print("The shape of the clean dataframe is:",to_be_saved.shape)

    print("Saving final clean data to:",file_name)
    to_be_saved.to_csv(file_name)
    print("Saved the clean dataframe at:",file_name)

