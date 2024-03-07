
# Calling required libraries

#Models
##Classification
from sklearn.svm import SVC

##Model frameworks
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn import preprocessing



#Utility
from os import makedirs as mkd
from tqdm import tqdm
from glob import glob


import pandas as pd
import numpy as np

import argparse
import pickle
import json

##########  Start--Function definition--##############
def fetch_data(exp_fname='GSE162694_norm.csv',gene_fname='FDR_list_cutoff_old.05.xlsx',cat_col="labels"):
	#Three input arguments required
	# 1. exp_fname: file path of main data file, having samples in rows and features in columns
	# 2. gene_fname: file path of feature subset or complete feature (if subset is not required)
	# 3. cat_cols: Name of colum in exmp_fname having class values or label information

    data=pd.read_csv(exp_fname,index_col=0)
    high_gene=pd.read_csv(gene_fname)
    ready_data=data[high_gene['Genes'].to_list()+[cat_col]]


    return ready_data


def feature_extract(model):
	#Required input: trained linear model

    importance = model.ranking_
    feature_importances = pd.Series(importance, index=model.feature_names_in_)
    sel_indx=model.support_
    my_feat=feature_importances[sel_indx]
    return my_feat,feature_importances

##########  End--Function definition--##############



##########  1. Start--Preparation required variables--##############
# Reading command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--inp_file',type=str)
parser.add_argument('--mri_type',type=str,default ='flair')
#parser.add_argument('--res_file',type=str)
parser.add_argument('--step_size',type=int,default=1)
parser.add_argument('--status',type=int,default=1)
args = parser.parse_args()

#loading configuration/config file
with open(args.inp_file) as f:
    configs = json.load(f)

#Initialize variables using config file and command line args
cat1 = configs['cat1']
cat2 = configs['cat2']
cat_col = configs['cat_col']
inp_src = configs['inp_src']
out_src = configs['out_src']
exp_fname = configs['exp_fname']
rnd = configs['rnd_seed']
f_title = configs['fig_title']
gene_fname = configs['gene_fname']
#f_name = args.res_file
step_size = args.step_size
status = args.status
mri_type=args.mri_type
#Initializing label encoder
le = preprocessing.LabelEncoder()
print('This is encoding--',cat1,'--',cat2)
le.fit([cat1, cat2])

exp_fname = f'{inp_src}final_mat_{mri_type}.csv'
f_name=f'{gene_fname.split("/")[-1][:-4]}_{mri_type}'

output_file=f'{inp_src}RFE_forest_step{step_size}_{f_name}.csv'#pickle file for result dumping

##########  End--Preparation required variables--##############





##########  2. Start--Data loading and preparation--##############
dataset = fetch_data(exp_fname,gene_fname,cat_col)

X=dataset.iloc[:,0:-1]
print("Dimension of input data ",X.shape)
labels=dataset.iloc[:,-1]
y=le.transform(labels)
##########  End--Data loading and preparation--##############


##########  3. Start--RFE method--##############
results={}

#model = RFECV(estimator=SVC(kernel='linear',random_state=rnd),step=step_size,cv=10,n_jobs=10,verbose=status)
model = RFECV(estimator=RandomForestClassifier(random_state=rnd,n_estimators=50),step=step_size,cv=10,n_jobs=10,verbose=status)
np.random.seed(rnd)
rfecv=model.fit(X,y)
#name=str(Model.estimator_).split('(')[0]

my_feat,all_feat=feature_extract(rfecv)
all_ranks=pd.concat([all_feat.sort_values().reset_index(),pd.DataFrame({'Std':rfecv.cv_results_['std_test_score']}),pd.DataFrame({'Mean':rfecv.cv_results_['mean_test_score']})],axis=1)
all_ranks.rename(columns={'index':'Genes','0':'Rank'},inplace=True)
all_ranks.to_csv(output_file,index=False)

##########  End--RFE method--##############





