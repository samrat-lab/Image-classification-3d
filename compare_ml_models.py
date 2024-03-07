#!/usr/bin/env python
# coding: utf-8

# In[1]:


# the libraries we need

#Models

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.svm import SVC


#Metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn import preprocessing




#Utility
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from glob import glob
from os import makedirs as mkd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import pandas as pd
import numpy as np

#import imblearn
import time
import pickle
import argparse
import json
import warnings


# In[2]:

warnings.filterwarnings("ignore", message=".*the default evaluation metric used with the objective.*")


# In[4]:


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def fetch_data(exp_fname='GSE162694_norm.csv',gene_fname='FDR_list_cutoff_old.05.xlsx',cat_col="labels"):

    data=pd.read_csv(exp_fname,index_col=0)

  
    high_gene=pd.read_csv(gene_fname)
    #print(data.shape,len(high_gene))
    ready_data=data[high_gene['Genes'].to_list()+[cat_col]]
    return ready_data


def feature_extract(model):
    feature_importances=[]
    importance = model.ranking_
    feature_importances = pd.Series(importance, index=model.feature_names_in_)
    sel_indx=model.support_
    my_feat=feature_importances[sel_indx]
    return my_feat,feature_importances


# In[5]:


def get_models(rnd=24):
    models = dict()
    # lr
    #rfe = RFECV(estimator=DecisionTreeClassifier(random_state=24),step=10,cv=10,n_jobs=25)
    model = DecisionTreeClassifier(random_state=rnd)
    models['dt'] = model
    # perceptron
    #rfe = RFECV(estimator=RandomForestClassifier(random_state=rnd,n_estimators=50),cv=10,n_jobs=25)
    model = RandomForestClassifier(random_state=rnd,n_estimators=50)
    models['rf'] = model
    # cart
    #KNN
    #rfe = RFECV(estimator=Perceptron(),cv=10,n_jobs=25)
    model = KNeighborsClassifier()
    models['knn'] = model
    # rf
    #rfe = RFECV(estimator=SVC(kernel='linear',random_state=rnd),cv=10,n_jobs=25)
    # model = SVC(kernel='linear',random_state=rnd)
    # models['svm'] = model
    # gbm
    #rfe = RFECV(estimator=XGBClassifier(random_state=rnd),cv=10,n_jobs=25)
    #model = XGBClassifier(use_label_encoder=False, random_state=rnd)
    #models['xgb'] = model
    #Some liner
    #rfe = RFECV(estimator=LogisticRegression(solver='lbfgs',max_iter=1000,random_state=rnd),cv=10,n_jobs=25)
    model = LogisticRegression(solver='lbfgs',max_iter=1000,random_state=rnd)
    models['logit'] = model
    return models


# In[6]:





def get_viz(inp_df,title,grp='test'):
    labels={'dt':'Decision Tree','rf':'Random Forest','knn':'KNN','svm':'SVM','xgb':'XGBoost','logit':'Logistic Regression'}
    inp_df.replace({"Algo":labels},inplace=True)
    viz_df=inp_df.filter(like=grp)
    viz_df.loc[:,'Algo']=inp_df['Algo']
    fig = px.box(
        viz_df,color='Algo',title=title,
        labels={}
                )
    #fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
    #fig.update_xaxes(tickvals=['Accuracy','F1-Score','AUROC','Precision','Recall'])
    fig.update_layout(
        xaxis_title=None,
        yaxis_title='Scores',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0,1, 2, 3, 4],
            ticktext = ['Accuracy','F1-Score','AUROC','Precision','Recall']
        ),
        font = dict(
            size=24,
            color='black'
            #family='Arial Black'
        ),
        #template='simple_white',
        width=1000,
        height=600,
        legend=dict(
        orientation="h",    
        title='Legends'
        )
    )
    return fig


def get_res_summary(FDR_res,f_name,fig_title='High Gene List'):
    #f_name='High_full_compare'
    output_file=f'{out_src}model_comparision{f_name}_avg.xlsx'
    #my_df=pd.DataFrame([])
    temp_df=[]
    for name,my_entry in FDR_res.items():
        df=pd.DataFrame(my_entry)
        df['Algo']=name
        temp_df.append(df)
        #my_df=my_df.append(df)
    my_df = pd.concat(temp_df)
    my_df.drop(['fit_time','score_time','estimator'],axis=1,inplace=True)
    with pd.ExcelWriter(output_file) as writer:

        df_mean=my_df.groupby('Algo').mean()
        df_std=my_df.groupby('Algo').std()
        df_mean.to_excel(writer,sheet_name='mean')
        df_std.to_excel(writer,sheet_name='std')
    #my_df.head()
    #fig = px.box(my_df,color='Algo',title=fig_title)
    return_fig2  = get_viz(my_df,f'{fig_title} Test dataset',grp='test')
    return_fig1  = get_viz(my_df,f'{fig_title} Train dataset',grp='train')
    

    return return_fig1, return_fig2

    

##########--Start--Preparation required variables ##############
parser = argparse.ArgumentParser()
parser.add_argument('--inp_file',type=str)
parser.add_argument('--fig_file',type=bool,default=False)
parser.add_argument('--mri_type',type=str,default ='flair')
#parser.add_argument('--gene_file',type=str)
#parser.add_argument('--res_file',type=str)
parser.add_argument('--rnd_val',type=int,default=0)
parser.add_argument('--n_cpu',type=int,default=6)
args = parser.parse_args()

with open(args.inp_file) as f:
    configs = json.load(f)

cat1 = configs['cat1']
cat2 = configs['cat2']
cat_col = configs['cat_col']
inp_src = configs['inp_src']
out_src = configs['out_src']
#f_name = configs['res_file']
#exp_fname = configs['exp_fname']
gene_fname = configs['gene_fname']
rnd = configs['rnd_seed']
f_title = configs['fig_title']
#gene_fname = args.gene_file
#f_name = args.res_file
#exp_fname = args.exp_file
#rnd = args.rnd_val
mri_type = args.mri_type
n_cpu=args.n_cpu


exp_fname = f'{inp_src}final_mat_{mri_type}.csv'


le = preprocessing.LabelEncoder()
print('This is encoding--',cat1,'--',cat2)
le.fit([cat1, cat2])

# files=glob(inp_src+'*csv')
# files
mkd(out_src,exist_ok=True)
f_name=f'{gene_fname.split("/")[-1][:-4]}_{mri_type}'
output_file=f'{out_src}model_comparision_{f_name}.xlsx'

##########--End--Preparation required variables ##############
    

##########--Start--Data loading and preparation##############


dataset = fetch_data(exp_fname,gene_fname,cat_col)

X=dataset.iloc[:,0:-1]
print("Dimension of input data ",X.shape)
labels=dataset.iloc[:,-1]
y=le.transform(labels)

##########--End--Data loading and preparation##############




##########--Start--Data loading and preparation##############
models = get_models(rnd)

results={}
with pd.ExcelWriter(output_file) as writer:

    model_iterator = tqdm(models.items())
    for itm in model_iterator:
        name = itm[0]
        model =itm[1]
        model_iterator.set_postfix(Current_model = name)

        rkf=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=rnd)
        cross_out = cross_validate(model, X, y,return_train_score=True,return_estimator=True ,scoring=('accuracy','f1','roc_auc','precision','recall'),n_jobs=n_cpu ,cv=rkf)
        results[name]=cross_out
        result_df=pd.DataFrame(cross_out)
        result_df.to_excel(writer, sheet_name=name)
        
pickle.dump( results, open( f'{out_src}{f_name}.p', "wb" ) )

##########--Start--Data loading and preparation##############


# In[24]:


my_fig1, my_fig2 = get_res_summary(results,f_name,f_title)
if args.fig_file:
    pio.write_image(my_fig1,f'{out_src}train_model_comparison_{f_name}.png',scale=10)
    pio.write_image(my_fig2,f'{out_src}test_model_comparison_{f_name}.png',scale=10)
#my_fig.show()



# In[10]:

