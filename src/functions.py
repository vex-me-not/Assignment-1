
import numpy as np
import pandas as pd
import sys
import os
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import seaborn as sns


COLORS={
            'BayesianRidge':'lightslategrey',
            'ElasticNet':'lightseagreen',
            'SVR':'lightpink'
}

"""Classes used in training start here"""

class DefaultPredictor:
    def __init__(self):
        self.models={
            'BayesianRidge':BayesianRidge(),
            'ElasticNet':ElasticNet(),
            'SVR':SVR()
            }
    
    def train_models(self,x_train,y_train):
        trained={}
        
        for name,model in self.models.items():
            model.fit(x_train,y_train)
            trained[name]=model
        
        return trained


class Evaluator:
    
    def __init__(self,model,name="Dummy"):
        self.model=model
        self.name=name
        self.metrics={
            'RMSE':[],
            'MAE':[],
            'R2':[]
        }
        self.stats=None
        self.ci=95 # confidence interval with alpha=0.05

    def __compute_stats(self):
        low=(100-self.ci)/2

        return {
            metric: {
                'mean': np.mean(values),
                'median': np.median(values),
                f'CI_{self.ci}': (np.percentile(values, low),
                                  np.percentile(values, 100 - low)),
                'std': np.std(values)
            }
            for metric, values in self.metrics.items()
        }

    def evaluate(self,x,y,iters=500,random_state=42):
        np.random.seed(random_state)
        
        for i in range(iters):
            x_rs,y_rs = resample(x,y)
            y_pred = self.model.predict(x_rs)
            
            self.metrics['RMSE'].append(np.sqrt(mean_squared_error(y_rs,y_pred)))
            self.metrics['MAE'].append(mean_absolute_error(y_rs,y_pred))
            self.metrics['R2'].append(r2_score(y_rs,y_pred))
        
        self.stats = self.__compute_stats()
        
    def report(self):
        
        if not self.stats:
            raise ValueError("Must run evalute() first!")
        
        report=pd.DataFrame.from_dict(self.stats,orient='index')

        report[['CIlow','CIhigh']]=pd.DataFrame(report[f'CI_{95}'].to_list(),index=report.index)

        report=report.round(4)

        to_be_returned=report[['mean','std','median','CIlow','CIhigh']]

        return to_be_returned
        
class IO:
    def __init__(self,models_dir):
        if models_dir is None:
            raise ValueError("Directory of models must be given!")

        self.models_dir=Path(models_dir)
        os.makedirs(self.models_dir,exist_ok=True)

    def __gen_filename(self,name,suffix):
        parts=[name]
        if suffix:
            parts.append(suffix)

        return self.models_dir/f"{'_'.join(parts)}.pkl"
        # return self.models_dir + '/' + f"{'_'.join(parts)}"
    
    def save(self,model,name,suf=''):
        joblib.dump(model,self.__gen_filename(name,suffix=suf))

    def load(self,name,suf=''):
        return joblib.load(self.__gen_filename(name,suffix=suf))


"""Helper functions start here"""


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
    # scaled_bacteria=scale_data(bacteria,bacteria_names)

    frames=[bmi,age,sex_encoded,bacteria]

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
    # print('Statistics=%.3f, p=%.3f' % (stat, p))

    # interpret
    if p > 0.05:
        print(f'{name} Sample LOOKS Gaussian (fail to reject H0)')
        return 1
    else:
        print(f'{name} Sample does NOT look Gaussian (reject H0)')
        return 0

def rough_idea(data,name):
    print(f"--> {name} <--\n Mean:{np.mean(data)} \n Standard deviation: {np.std(data)}\n Max:{np.max(data)} \n Min:{np.min(data)}")
    return (np.mean(data),np.std(data),np.max(data),np.min(data))

def get_bacteria(df,mode="explore"):
    # we keep the bacteria only
    if mode=="explore":
        to_drop=['Unnamed: 0', 'Project ID','Experiment type','Sex','Host age','Disease MESH ID','BMI']
    else:
        to_drop=['Sex','Host age','BMI']
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
        elif mode=="Corr":
            print(f"Pearson's correlation of {key} and BMI is {value.statistic}")
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

def remove_key(d:dict, key):
    r = dict(d)
    del r[key]
    return r

def plot_bact_corr(frames):
    to_be_plotted=pd.concat(frames,axis=1)
    corr_mat=to_be_plotted.corr()
    
    sns.heatmap(corr_mat,annot=True)
    plt.show()

def viz_comparison(*evals,figsize=(18,6)):
    plt.figure(figsize=figsize)
    
    metrics={
        'RMSE':[],
        'MAE':[],
        'R2':[]
    }

    labels=[ev.name for ev in evals]

    for ev in evals:
        metrics['RMSE'].append(ev.metrics['RMSE'])
        metrics['MAE'].append(ev.metrics['MAE'])
        metrics['R2'].append(ev.metrics['R2'])

    for i, (name,values) in enumerate(metrics.items(),1):
        plt.subplot(1,3,i)
        boxplt=plt.boxplot(values,labels=labels,patch_artist=True,widths=0.6)
        
        for box,lbl in zip(boxplt['boxes'],labels):
            box.set_facecolor(COLORS.get(lbl,'gray'))
        
        plt.title(f"Comparison method: {name}")
        plt.ylabel("Score" if name=='R2' else 'Value of Error')
        plt.grid(True,alpha=0.3)
    
    plt.tight_layout()
    plt.show()