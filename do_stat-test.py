def common_ls(ls1,ls2):
    return list(set(ls1)&set(ls2))

def fdr(p_vals):
    ranked_p_values = stat_tool.rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

def stat_test(exp_data,label_column='MGMT_value',grp1=0,grp2=1,test_type='ttest'):
    #grp1='Group1'
    #grp2='Group2'
    
    #test_frame=exp_data[gene_names]
    test_frame = exp_data.drop([label_column],axis=1)
    case1=test_frame[exp_data[label_column]==grp1].values
    case2=test_frame[exp_data[label_column]==grp2].values
    print(case1.shape,case2.shape)
    mean_c=case1.mean(axis=0)
    mean_c2=case2.mean(axis=0)
    fold_change=mean_c2/mean_c
    if test_type == 'ttest':
        U1, p = stat_tool.ttest_ind(case1, case2)
    else:
        U1, p = stat_tool.mannwhitneyu(case1, case2, method="exact")
    
    man_test=pd.DataFrame({'Genes':test_frame.columns.to_list(),'P_value':p,'FDR':fdr(p),'Fold':fold_change})
    #print(man_test.head())
    #man_test.to_csv(out_file,index=False)
    return man_test


import scipy.stats as stat_tool
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mri_type',default='flair',type=str)
parser.add_argument('--test_type',default='ttest',type=str)
args = parser.parse_args()

mri_type = args.mri_type
test_name = args.test_type

meta_file =  '../Input/meta_data/train_labels.csv'
meta_data = pd.read_csv(meta_file)
meta_data.rename(columns={'BraTS21ID':'Pat_ID'},inplace=True)
#print(meta_data.shape)
#meta_data.head()


feat_df_flar = pd.read_csv(f'feature_file-{mri_type}.csv',index_col=0)
sel_cols = [itm for itm in feat_df_flar.columns if 'diagnostics' not in itm ]
#print(feat_df_flar.shape)
#feat_df_flar.head()


new_df = pd.merge(feat_df_flar[sel_cols],meta_data,on = 'Pat_ID',how = 'inner').set_index('Pat_ID')
#print(new_df.shape)
#new_df.head()

new_df_test = stat_test(new_df,test_type=test_name)


new_df_test.to_csv(f'significance_{test_name}_{mri_type}.csv',index=None)