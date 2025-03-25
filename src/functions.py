
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

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
    bacteria=get_bacteria(data_df)

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

def plot_for_normality(data):
    #histogram
    plt.hist(data)
    # qqplot
    sm.qqplot(data, line='s')
    plt.show()


def check_for_normality(data,name):

    stat, p = stats.shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # interpret
    if p > 0.05:
        print(f'{name} Sample looks Gaussian (fail to reject H0)')
        return 1
    else:
        print(f'{name} Sample does not look Gaussian (reject H0)')
        return 0

def rough_idea(data,name):
    # print(f"--> {name} <--\n Mean:{np.mean(data)} \n Standard deviation: {np.std(data)}\n Max:{np.max(data)} \n Min:{np.min(data)}")
    return (np.mean(data),np.std(data),np.max(data),np.min(data))

def get_bacteria(df):
        # we keep the bacteria only
    to_drop=['Unnamed: 0', 'Project ID','Experiment type','Sex','Host age','Disease MESH ID','BMI']
    bacteria=df.drop(labels=to_drop,axis=1)
    return bacteria

def print_dict(dict,mode=None):

    for key,value in dict.items():
        if mode=="Normal":
            if value==0:
                print(f"{key} IS NOT normally distrubuted")
            else:
                print(f"{key} IS normally distrubuted")
        elif mode=="Mean":
            print(f"Mean of {key} is {value}")
        elif mode=="Std":
            print(f"Standard deviation of {key} is {value}")
        elif mode=="Max":
            print(f"Max of {key} is {value}")
        elif mode=="Min":
            print(f"Min of {key} is {value}")
        else:
            print(f"{key} --> {value}")


def find_normal(normality_dict):
    normal_columns=[]
    non_normal_columns=[]
    for key,value in normality_dict.items():
        if value==0:
            non_normal_columns.append(key)
        else:
            normal_columns.append(key)
    return [normal_columns,non_normal_columns]