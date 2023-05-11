import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import scipy.stats as stats
import random
from utils import load_boolq_formatted_dataset, load_squad_formatted_dataset
from rajpurkar_squad2_evaluate import compute_f1 #the squad2.0 evaluation code, available at https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

#where to save the figures
fig_save_dir = r'figures'

#what seeds are we using in our models?
seeds=[2, 17, 42, 97, 136]

#epileptologist dataset paths
epi_hasSz_dataset_path = 'penn_generalization_eval_datasets/classification_epi.json'
epi_szFreq_dataset_path = 'penn_generalization_eval_datasets/eqa_epi.json'
#neurologist dataset paths
neuro_hasSz_dataset_path = 'penn_generalization_eval_datasets/classification_neuro.json'
neuro_szFreq_dataset_path = 'penn_generalization_eval_datasets/eqa_neuro.json'
#non-neurologist dataset directory
general_hasSz_dataset_path = 'penn_generalization_eval_datasets/classification_general.json'
general_szFreq_dataset_path = 'penn_generalization_eval_datasets/eqa_general.json'

#load the datasets
epi_hasSz_dataset = load_boolq_formatted_dataset(epi_hasSz_dataset_path)
epi_szFreq_dataset = load_squad_formatted_dataset(epi_szFreq_dataset_path)['data']
neuro_hasSz_dataset = load_boolq_formatted_dataset(neuro_hasSz_dataset_path)
neuro_szFreq_dataset = load_squad_formatted_dataset(neuro_szFreq_dataset_path)['data']
general_hasSz_dataset = load_boolq_formatted_dataset(general_hasSz_dataset_path)
general_szFreq_dataset = load_squad_formatted_dataset(general_szFreq_dataset_path)['data']

one_hot = {'Yes':1, 'No':0, 'no-answer': 2}
all_predictions = {}
overall_performance = {}

#merge predictions and dataset text
for seed in seeds:
    seed_predictions = {}

    #epileptologist prediction paths
    epi_hasSz_prediction_path = fr'model_generalization_predictions/hasSz_epi_NOTES_MODEL_{seed}/eval_predictions.tsv'
    epi_szFreq_prediction_path = fr'model_generalization_predictions/szFreq_epi_NOTES_MODEL_{seed}/eval_predictions.json'
    #neurologist prediction paths
    neuro_hasSz_prediction_path = fr'model_generalization_predictions/hasSz_neuro_NOTES_MODEL_{seed}/eval_predictions.tsv'
    neuro_szFreq_prediction_path = fr'model_generalization_predictions/szFreq_neuro_NOTES_MODEL_{seed}/eval_predictions.json'
    #non-neurologist prediction paths
    general_hasSz_prediction_path = fr'model_generalization_predictions/hasSz_general_NOTES_MODEL_{seed}/eval_predictions.tsv'
    general_szFreq_prediction_path = fr'model_generalization_predictions/szFreq_general_NOTES_MODEL_{seed}/eval_predictions.json'
    
    #load the predictions
    epi_hasSz_preds = pd.read_csv(epi_hasSz_prediction_path, sep='\t', header=0)
    with open(epi_szFreq_prediction_path, 'r') as f:
        epi_szFreq_preds = json.load(f)
    neuro_hasSz_preds = pd.read_csv(neuro_hasSz_prediction_path, sep='\t', header=0)
    with open(neuro_szFreq_prediction_path, 'r') as f:
        neuro_szFreq_preds = json.load(f)
    general_hasSz_preds = pd.read_csv(general_hasSz_prediction_path, sep='\t', header=0)
    with open(general_szFreq_prediction_path, 'r') as f:
        general_szFreq_preds = json.load(f)

    #epileptologists.
    for i in range(len(epi_hasSz_dataset)):
        #get the doc_id of the datum
        doc_id = "_".join(epi_hasSz_dataset[i]['id'].split("_")[:-1])
        if doc_id in seed_predictions: #there should be no overlap between the datasets
            print("WARNING: THIS EPI DOCUMENT IS NOT UNIQUE")

        #load basic info of the passage
        seed_predictions[doc_id] = {}
        seed_predictions[doc_id]['context'] = epi_hasSz_dataset[i]['passage']
        seed_predictions[doc_id]['type'] = 'epileptologist'

        #get the answers for hasSz
        seed_predictions[doc_id]['hasSz_gt'] = epi_hasSz_dataset[i]['label']

        #get the prediction for hasSz
        #verify that the ID is correct
        if "_".join(epi_hasSz_preds.iloc[i]['ID'].split("_")[:-1]) == doc_id:
            seed_predictions[doc_id]['hasSz_pred'] = epi_hasSz_preds.iloc[i]['argmax']
        else:
            print("WARNING, PREDICTION - DATASET INDEXING DESYNCHRONIZED | HasSz Data - HasSz Pred")

        #get the answers and predictions for pqf/elo
        for j in range(2):
            #check that the hasSz and SzFreq datasets line up
            if "_".join(epi_szFreq_dataset[2*i + j]['id'].split("_")[:-1]) != doc_id:
                print("WARNING, DATASET - DATASET INDEXING DESYNCHONIZED | SzFreq - HasSz")
            if 'pqf' in epi_szFreq_dataset[2*i + j]['id']:
                seed_predictions[doc_id]['pqf_gt'] = epi_szFreq_dataset[2*i + j]['answers']['text']
                seed_predictions[doc_id]['pqf_pred'] = epi_szFreq_preds[epi_szFreq_dataset[2*i + j]['id']]
            else:
                seed_predictions[doc_id]['elo_gt'] = epi_szFreq_dataset[2*i + j]['answers']['text']
                seed_predictions[doc_id]['elo_pred'] = epi_szFreq_preds[epi_szFreq_dataset[2*i + j]['id']]

    #neurologists.
    for i in range(len(neuro_hasSz_dataset)):
        #get the doc_id of the datum
        doc_id = "_".join(neuro_hasSz_dataset[i]['id'].split("_")[:-1])
        if doc_id in seed_predictions: #there should be no overlap between the datasets
            print("WARNING: THIS EPI DOCUMENT IS NOT UNIQUE")

        #load basic info of the passage
        seed_predictions[doc_id] = {}
        seed_predictions[doc_id]['context'] = neuro_hasSz_dataset[i]['passage']
        seed_predictions[doc_id]['type'] = 'neurologist'

        #get the answers for hasSz
        seed_predictions[doc_id]['hasSz_gt'] = neuro_hasSz_dataset[i]['label']

        #get the prediction for hasSz
        #verify that the ID is correct
        if "_".join(neuro_hasSz_preds.iloc[i]['ID'].split("_")[:-1]) == doc_id:
            seed_predictions[doc_id]['hasSz_pred'] = neuro_hasSz_preds.iloc[i]['argmax']
        else:
            print("WARNING, PREDICTION - DATASET INDEXING DESYNCHRONIZED | HasSz Data - HasSz Pred")

        #get the answers and predictions for pqf/elo
        for j in range(2):
            #check that the hasSz and SzFreq datasets line up
            if "_".join(neuro_szFreq_dataset[2*i + j]['id'].split("_")[:-1]) != doc_id:
                print("WARNING, DATASET - DATASET INDEXING DESYNCHONIZED | SzFreq - HasSz")
            if 'pqf' in neuro_szFreq_dataset[2*i + j]['id']:
                seed_predictions[doc_id]['pqf_gt'] = neuro_szFreq_dataset[2*i + j]['answers']['text']
                seed_predictions[doc_id]['pqf_pred'] = neuro_szFreq_preds[neuro_szFreq_dataset[2*i + j]['id']]
            else:
                seed_predictions[doc_id]['elo_gt'] = neuro_szFreq_dataset[2*i + j]['answers']['text']
                seed_predictions[doc_id]['elo_pred'] = neuro_szFreq_preds[neuro_szFreq_dataset[2*i + j]['id']]

    #non-neurologists.
    for i in range(len(general_hasSz_dataset)):
        #get the doc_id of the datum
        doc_id = "_".join(general_hasSz_dataset[i]['id'].split("_")[:-1])
        if doc_id in seed_predictions: #there should be no overlap between the datasets
            print("WARNING: THIS EPI DOCUMENT IS NOT UNIQUE")

        #load basic info of the passage
        seed_predictions[doc_id] = {}
        seed_predictions[doc_id]['context'] = general_hasSz_dataset[i]['passage']
        seed_predictions[doc_id]['type'] = 'generalist'

        #get the answers for hasSz
        seed_predictions[doc_id]['hasSz_gt'] = general_hasSz_dataset[i]['label']

        #get the prediction for hasSz
        #verify that the ID is correct
        if "_".join(general_hasSz_preds.iloc[i]['ID'].split("_")[:-1]) == doc_id:
            seed_predictions[doc_id]['hasSz_pred'] = general_hasSz_preds.iloc[i]['argmax']
        else:
            print("WARNING, PREDICTION - DATASET INDEXING DESYNCHRONIZED | HasSz Data - HasSz Pred")

        #get the answers and predictions for pqf/elo
        for j in range(2):
            #check that the hasSz and SzFreq datasets line up
            if "_".join(general_szFreq_dataset[2*i + j]['id'].split("_")[:-1]) != doc_id:
                print("WARNING, DATASET - DATASET INDEXING DESYNCHONIZED | SzFreq - HasSz")
            if 'pqf' in general_szFreq_dataset[2*i + j]['id']:
                seed_predictions[doc_id]['pqf_gt'] = general_szFreq_dataset[2*i + j]['answers']['text']
                seed_predictions[doc_id]['pqf_pred'] = general_szFreq_preds[general_szFreq_dataset[2*i + j]['id']]
            else:
                seed_predictions[doc_id]['elo_gt'] = general_szFreq_dataset[2*i + j]['answers']['text']
                seed_predictions[doc_id]['elo_pred'] = general_szFreq_preds[general_szFreq_dataset[2*i + j]['id']]

    #convert to a dataframe
    seed_predictions = pd.DataFrame(seed_predictions).transpose()
    
    #calculate accuracy and F1 scores
    performance = {}
    for idx, row in seed_predictions.iterrows():
        performance[idx] = {}

        #hasSz accuracy is binary, 1 for correct
        performance[idx]['hasSz_acc'] = 1 if row['hasSz_gt'] == row['hasSz_pred'] else 0

        #get pqf and elo f1s
        pqf_f1s = [compute_f1(gt, row['pqf_pred']) for gt in row['pqf_gt']] #get the f1 score if there is a ground truth answer
        if pqf_f1s:
            performance[idx]['pqf_f1_hasAns'] = np.max(pqf_f1s)
        else: #if there wasn't a grund truth answer, then check that the prediction is blank
            performance[idx]['pqf_f1_noAns'] = float(row['pqf_pred'] == '')
        elo_f1s = [compute_f1(gt, row['elo_pred']) for gt in row['elo_gt']] #get the f1 score if there is a ground truth answer
        if elo_f1s:
            performance[idx]['elo_f1_hasAns'] = np.max(elo_f1s)
        else: #if there wasn't a grund truth answer, then check that the prediction is blank
            performance[idx]['elo_f1_noAns'] = float(row['elo_pred'] == '')
            
    #add the performance scores to the master table
    seed_predictions = seed_predictions.merge(pd.DataFrame(performance).transpose(), left_index=True, right_index=True).reset_index() #reset index for easier indexing in the next cell
    
    all_predictions[seed] = seed_predictions
    
    
    #calculate overall performance scores
    overall_performance[seed] = {'epi_hasAns_Acc':np.nanmean(seed_predictions.loc[seed_predictions['type']=='epileptologist', 'hasSz_acc']),
                                'neuro_hasAns_Acc':np.nanmean(seed_predictions.loc[seed_predictions['type']=='neurologist', 'hasSz_acc']),
                                'gen_hasAns_Acc':np.nanmean(seed_predictions.loc[seed_predictions['type']=='generalist', 'hasSz_acc']),
                                'epi_PQF_F1_Overall':np.mean(np.nanmax(seed_predictions.loc[seed_predictions['type']=='epileptologist', ['pqf_f1_noAns', 'pqf_f1_hasAns']], axis=1)),
                                'neuro_PQF_F1_Overall':np.mean(np.nanmax(seed_predictions.loc[seed_predictions['type']=='neurologist', ['pqf_f1_noAns', 'pqf_f1_hasAns']], axis=1)),
                                'gen_PQF_F1_Overall':np.mean(np.nanmax(seed_predictions.loc[seed_predictions['type']=='generalist', ['pqf_f1_noAns', 'pqf_f1_hasAns']], axis=1)),
                                'epi_ELO_F1_Overall':np.mean(np.nanmax(seed_predictions.loc[seed_predictions['type']=='epileptologist', ['elo_f1_noAns', 'elo_f1_hasAns']], axis=1)),
                                'neuro_ELO_F1_Overall':np.mean(np.nanmax(seed_predictions.loc[seed_predictions['type']=='neurologist', ['elo_f1_noAns', 'elo_f1_hasAns']], axis=1)),
                                'gen_ELO_F1_Overall':np.mean(np.nanmax(seed_predictions.loc[seed_predictions['type']=='generalist', ['elo_f1_noAns', 'elo_f1_hasAns']], axis=1)),
                                }
    
    
    
classification_performance = np.array([[overall_performance[seed][f'{prov_type}_hasAns_Acc'] for prov_type in ['epi', 'neuro', 'gen']] for seed in seeds])


#go through and find the sizes of each answer category
#hasSz: yes, no, IDK
epi_szFree_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'epileptologist') & (seed_predictions['hasSz_gt'] == 0)])
epi_hasSz_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'epileptologist') & (seed_predictions['hasSz_gt'] == 1)])
epi_idk_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'epileptologist') & (seed_predictions['hasSz_gt'] == 2)])
neuro_szFree_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'neurologist') & (seed_predictions['hasSz_gt'] == 0)])
neuro_hasSz_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'neurologist') & (seed_predictions['hasSz_gt'] == 1)])
neuro_idk_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'neurologist') & (seed_predictions['hasSz_gt'] == 2)])
gen_szFree_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'generalist') & (seed_predictions['hasSz_gt'] == 0)])
gen_hasSz_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'generalist') & (seed_predictions['hasSz_gt'] == 1)])
gen_idk_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'generalist') & (seed_predictions['hasSz_gt'] == 2)])

#PQF: Exists, IDK = 1 - Exist
epi_pqf_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'epileptologist') & (seed_predictions['pqf_gt'].astype(bool))])
neuro_pqf_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'neurologist') & (seed_predictions['pqf_gt'].astype(bool))])
gen_pqf_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'generalist') & (seed_predictions['pqf_gt'].astype(bool))])

#ELO: Exists, IDK = 1 - Exist
epi_elo_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'epileptologist') & (seed_predictions['elo_gt'].astype(bool))])
neuro_elo_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'neurologist') & (seed_predictions['elo_gt'].astype(bool))])
gen_elo_ct = len(seed_predictions.loc[(seed_predictions['type'] == 'generalist') & (seed_predictions['elo_gt'].astype(bool))])


#number of each author type
num_epi = len(seed_predictions.loc[(seed_predictions['type'] == 'epileptologist')])
num_neuro = len(seed_predictions.loc[(seed_predictions['type'] == 'neurologist')])
num_gen = len(seed_predictions.loc[(seed_predictions['type'] == 'generalist')])



#construct confusion matrix for hasSz across the seeds
def initialize_conf_mat():
    hasSz_mat = {}
    for key in ['true_0', 'true_1', 'true_2']:
        hasSz_mat[key] = {}
        for val in ['pred_0', 'pred_1', 'pred_2']:
            hasSz_mat[key][val] = list()
    return hasSz_mat
epi_mat = initialize_conf_mat()
neuro_mat = initialize_conf_mat()
gen_mat = initialize_conf_mat()
epi_mat_means = initialize_conf_mat()
neuro_mat_means = initialize_conf_mat()
gen_mat_means = initialize_conf_mat()
epi_mat_labels = initialize_conf_mat()
neuro_mat_labels = initialize_conf_mat()
gen_mat_labels = initialize_conf_mat()

#go through each seed
for seed in all_predictions:
    #first, add to the confusion matrix for epileptologists
    results = all_predictions[seed].loc[(all_predictions[seed]['type'] == 'epileptologist')]
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            epi_mat[f"true_{i}"][f"pred_{j}"].append(len(results.loc[(results['hasSz_gt'] == i) & (results['hasSz_pred'] == j)]))
            
    #then, add to the confusion matrix for neurologists
    test = all_predictions[seed].loc[(all_predictions[seed]['type'] == 'neurologist')]
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            neuro_mat[f"true_{i}"][f"pred_{j}"].append(len(test.loc[(test['hasSz_gt'] == i) & (test['hasSz_pred'] == j)]))
            
    #finally, add to the confusion matrix for non-neurologists
    results = all_predictions[seed].loc[(all_predictions[seed]['type'] == 'generalist')]
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            gen_mat[f"true_{i}"][f"pred_{j}"].append(len(results.loc[(results['hasSz_gt'] == i) & (results['hasSz_pred'] == j)]))
            

#for each matrix, convert the vector into means 
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        #for the means value only
        epi_mat_means[f"true_{i}"][f"pred_{j}"] = np.mean(epi_mat[f"true_{i}"][f"pred_{j}"])
        neuro_mat_means[f"true_{i}"][f"pred_{j}"] = np.mean(neuro_mat[f"true_{i}"][f"pred_{j}"])
        gen_mat_means[f"true_{i}"][f"pred_{j}"] = np.mean(gen_mat[f"true_{i}"][f"pred_{j}"])
        
#convert everything to pandas dataframe
#normalize dataframe by the sum of the columns
#this tells us what proportion of true X was actually calculated as pred X
epi_mat_means = pd.DataFrame(epi_mat_means).apply(lambda x: x/(np.sum(x)))
neuro_mat_means = pd.DataFrame(neuro_mat_means).apply(lambda x: x/(np.sum(x)))
gen_mat_means = pd.DataFrame(gen_mat_means).apply(lambda x: x/(np.sum(x)))
        
#for each matrix, convert the vector into mean 
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        #for the labels of the heatmap
        epi_mat_labels[f"true_{i}"][f"pred_{j}"] = f'{np.mean(epi_mat[f"true_{i}"][f"pred_{j}"])}\n[{np.min(epi_mat[f"true_{i}"][f"pred_{j}"])} - {np.max(epi_mat[f"true_{i}"][f"pred_{j}"])}]'
        neuro_mat_labels[f"true_{i}"][f"pred_{j}"] = f'{np.mean(neuro_mat[f"true_{i}"][f"pred_{j}"])}\n[{np.min(neuro_mat[f"true_{i}"][f"pred_{j}"])} - {np.max(neuro_mat[f"true_{i}"][f"pred_{j}"])}]'
        gen_mat_labels[f"true_{i}"][f"pred_{j}"] = f'{np.mean(gen_mat[f"true_{i}"][f"pred_{j}"])}\n[{np.min(gen_mat[f"true_{i}"][f"pred_{j}"])} - {np.max(gen_mat[f"true_{i}"][f"pred_{j}"])}]'
        
#convert everything to pandas dataframe
epi_mat_labels = pd.DataFrame(epi_mat_labels)
neuro_mat_labels = pd.DataFrame(neuro_mat_labels)
gen_mat_labels = pd.DataFrame(gen_mat_labels)


#======================calculate mannwhitneyu tests======================#
#classification - Epi vs Neuro, Epi vs. Non, Neuro vs Non
class_Epi_Neuro_stat, class_Epi_Neuro_p = stats.mannwhitneyu(
    classification_performance[:, 0], classification_performance[:, 1])
class_Epi_Non_stat, class_Epi_Non_p = stats.mannwhitneyu(
    classification_performance[:, 0], classification_performance[:, 2])
class_Neuro_Non_stat, class_Neuro_Non_p = stats.mannwhitneyu(
    classification_performance[:, 1], classification_performance[:, 2])
print(f"""For Classification between Epileptologists, Neurologists, and Non-Neurologists,
      Epileptologist == Neurologist p-value = {class_Epi_Neuro_p},
      Epileptologist == Non-Neurologist p-value = {class_Epi_Non_p},
      Neurologist == Non-Neurologist p-value = {class_Neuro_Non_p}
      """)
print("")

#PQF_overall - Epi vs Neuro, Epi vs. Non, Neuro vs Non
pqf_Epi_Neuro_stat, pqf_Epi_Neuro_p = stats.mannwhitneyu(
    pqf_performance[:, 0], pqf_performance[:, 1])
pqf_Epi_Non_stat, pqf_Epi_Non_p = stats.mannwhitneyu(
    pqf_performance[:, 0], pqf_performance[:, 2])
pqf_Neuro_Non_stat, pqf_Neuro_Non_p = stats.mannwhitneyu(
    pqf_performance[:, 1], pqf_performance[:, 2])
print(f"""For EQA Overall between Epileptologists, Neurologists, and Non-Neurologists,
      Epileptologist == Neurologist p-value = {pqf_Epi_Neuro_p},
      Epileptologist == Non-Neurologist p-value = {pqf_Epi_Non_p},
      Neurologist == Non-Neurologist p-value = {pqf_Neuro_Non_p}
      """)
print("")

#ELO_overall - Epi vs Neuro, Epi vs. Non, Neuro vs Non
elo_Epi_Neuro_stat, elo_Epi_Neuro_p = stats.mannwhitneyu(
    elo_performance[:, 0], elo_performance[:, 1])
elo_Epi_Non_stat, elo_Epi_Non_p = stats.mannwhitneyu(
    elo_performance[:, 0], elo_performance[:, 2])
elo_Neuro_Non_stat, elo_Neuro_Non_p = stats.mannwhitneyu(
    elo_performance[:, 1], elo_performance[:, 2])
print(f"""For EQA HasAns between Epileptologists, Neurologists, and Non-Neurologists,
      Epileptologist == Neurologist p-value = {elo_Epi_Neuro_p},
      Epileptologist == Non-Neurologist p-value = {elo_Epi_Non_p},
      Neurologist == Non-Neurologist p-value = {elo_Neuro_Non_p}
      """)
print("")
#============================================#

#======================Plot performance======================#
#boxplot data
results_classification = [classification_performance[:,0], classification_performance[:,1], classification_performance[:,2]]
results_eqa = [pqf_performance[:,0], pqf_performance[:,1], pqf_performance[:,2],
             elo_performance[:,0], elo_performance[:,1], elo_performance[:,2]]
results = results_classification + results_eqa
positions = [1, 1.5, 2,
             3.5, 4, 4.5,
             6, 6.5, 7]
#parameters for brackets for significant differences, and colors for dots
significance = 0.01667
colors = ['#B50A49', '#1E88E5', '#FFC107']
lower_anno_raise = 0.03
upper_anno_raise = 0.08
#https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
legend_elements = [Line2D([0], [0], color='k', marker='o', label='Epileptologist Notes', markerfacecolor=colors[0]),
                   Line2D([0], [0], color='k', marker='o', label='Neurologist Notes', markerfacecolor=colors[1]),
                   Line2D([0], [0], color='k', marker='o', label='Non-Neurologist Notes', markerfacecolor=colors[2]),]
                   #Line2D([0], [0], color='k', marker='*', label=f'{round(significance*100, 2)}% Significant', markerfacecolor='k', linewidth=0)]

#plot classification results
fig, axs = plt.subplots(1, 1, dpi=600, figsize=(8,4))
sns.despine()
#plot the boxplot
axs.boxplot(results, sym="", positions=positions)
#plot the individual points
axs.plot(np.ones(5)*positions[0], classification_performance[:,0], '.', color=colors[0], alpha=0.75)
axs.plot(np.ones(5)*positions[1], classification_performance[:,1], '.', color=colors[1], alpha=0.75)
axs.plot(np.ones(5)*positions[2], classification_performance[:,2], '.', color=colors[2], alpha=0.75)
#PQF performance
axs.plot(np.ones(5)*positions[3], pqf_performance[:,0], '.', color=colors[0], alpha=0.75)
axs.plot(np.ones(5)*positions[4], pqf_performance[:,1], '.', color=colors[1], alpha=0.75)
axs.plot(np.ones(5)*positions[5], pqf_performance[:,2], '.', color=colors[2], alpha=0.75)
#ELO Performance
axs.plot(np.ones(5)*positions[6], elo_performance[:,0], '.', color=colors[0], alpha=0.75)
axs.plot(np.ones(5)*positions[7], elo_performance[:,1], '.', color=colors[1], alpha=0.75)
axs.plot(np.ones(5)*positions[8], elo_performance[:,2], '.', color=colors[2], alpha=0.75)
axs.set_xticks(ticks=[1.5, 4, 6.5])
axs.set_xticklabels(labels=[f'Seizure Freedom\nClassification (Agreement)', f"Seizure Frequency\nExtraction (F$_1$)", f"Date of Last Seizure\nExtraction (F$_1$)"])

#classification significances
if class_Epi_Neuro_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[0], positions[1]]), 
                     np.max([classification_performance[:, 0], classification_performance[:, 1]])+ lower_anno_raise],
                 xytext=[np.mean([positions[0], positions[1]]), 
                         np.max([classification_performance[:, 0], classification_performance[:, 1]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.25', lw=1))
if class_Epi_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[0], positions[2]]), 
                     np.max([classification_performance[:, 0], classification_performance[:, 1]]) + upper_anno_raise],
                 xytext=[np.mean([positions[0], positions[2]]), 
                         np.max([classification_performance[:, 0], classification_performance[:, 1]]) + upper_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=3.05, lengthB=0.25', lw=1))
if class_Neuro_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[1], positions[2]]), 
                     np.max([classification_performance[:, 0], classification_performance[:, 1]])+ lower_anno_raise],
                 xytext=[np.mean([positions[1], positions[2]]), 
                         np.max([classification_performance[:, 0], classification_performance[:, 1]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.25', lw=1))



#pqf significances
if pqf_Epi_Neuro_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[3], positions[4]]), 
                     np.max([pqf_performance[:, 0], pqf_performance[:, 1], pqf_performance[:, 2]])+ lower_anno_raise],
                 xytext=[np.mean([positions[3], positions[4]]), 
                         np.max([pqf_performance[:, 0], pqf_performance[:, 1], pqf_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.25', lw=1))
if pqf_Epi_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[3], positions[5]]), 
                     np.max([pqf_performance[:, 0], pqf_performance[:, 1], pqf_performance[:, 2]]) + upper_anno_raise],
                 xytext=[np.mean([positions[3], positions[5]]), 
                         np.max([pqf_performance[:, 0], pqf_performance[:, 1], pqf_performance[:, 2]]) + upper_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=3.05, lengthB=0.25', lw=1))
if pqf_Neuro_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[4], positions[5]]), 
                     np.max([pqf_performance[:, 0], pqf_performance[:, 1], pqf_performance[:, 2]])+ lower_anno_raise],
                 xytext=[np.mean([positions[4], positions[5]]), 
                         np.max([pqf_performance[:, 0], pqf_performance[:, 1], pqf_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.25', lw=1))



#elo significances
if elo_Epi_Neuro_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[6], positions[7]]), 
                     np.max([elo_performance[:, 0], elo_performance[:, 1], elo_performance[:, 2]])+ lower_anno_raise],
                 xytext=[np.mean([positions[6], positions[7]]), 
                         np.max([elo_performance[:, 0], elo_performance[:, 1], elo_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.25', lw=1))
if elo_Epi_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[6], positions[8]]), 
                     np.max([elo_performance[:, 0], elo_performance[:, 1], elo_performance[:, 2]]) + upper_anno_raise],
                 xytext=[np.mean([positions[6], positions[8]]), 
                         np.max([elo_performance[:, 0], elo_performance[:, 1], elo_performance[:, 2]]) + upper_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=3.05, lengthB=0.25', lw=1))
if elo_Neuro_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[7], positions[8]]), 
                     np.max([elo_performance[:, 0], elo_performance[:, 1], elo_performance[:, 2]])+ lower_anno_raise],
                 xytext=[np.mean([positions[7], positions[8]]), 
                         np.max([elo_performance[:, 0], elo_performance[:, 1], elo_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.25', lw=1))
axs.set_ylabel('Agreement or F$_1$')
axs.set_ylim([0.35, 1.1])
axs.grid(True, alpha=0.25)
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=False)
plt.show()
fig.savefig(f"{fig_save_dir}/fig_1A.png", dpi=600, bbox_inches='tight')
fig.savefig(f"{fig_save_dir}/fig_1A.pdf", dpi=600, bbox_inches='tight')

# =============================================================================#

#======================Plot performance stratified by if answer exists======================#
#HasAns performance
#Column order: Epileptologists, Neurologists, NonNeurologists
#row order: seed 2, 17, 42, 97, 136
eqa_hasAns_performance = np.array([
    [77.8175,   64.0010, 61.7150],
    [74.7527,   53.8714, 65.0932],
    [72.8550,   62.2827, 61.6103],
    [74.6873,   64.7272, 60.8389],
    [74.9121,   65.1319, 62.4879]
    ])/100

#noAns performance
#Column order: Epileptologists, Neurologists, NonNeurologists
#row order: seed 2, 17, 42, 97, 136
eqa_noAns_performance = np.array([
    [90.4624,	89.8551,   92.3567],
    [93.6416,	91.3043,   94.9045],
    [92.4855,	92.0290,   92.9936],
    [90.1734,	89.1304,   91.7197],
    [93.0636,	90.5797,   91.0828]
    ])/100

#EQA_hasAns - Epi vs Neuro, Epi vs. Non, Neuro vs Non
eqa_hasAns__Epi_Neuro_stat, eqa_hasAns__Epi_Neuro_p = stats.mannwhitneyu(
    eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1])
eqa_hasAns__Epi_Non_stat, eqa_hasAns__Epi_Non_p = stats.mannwhitneyu(
    eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 2])
eqa_hasAns__Neuro_Non_stat, eqa_hasAns__Neuro_Non_p = stats.mannwhitneyu(
    eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2])
print(f"""For EQA HasAns between Epileptologists, Neurologists, and Non-Neurologists,
      Epileptologist == Neurologist p-value = {eqa_hasAns__Epi_Neuro_p},
      Epileptologist == Non-Neurologist p-value = {eqa_hasAns__Epi_Non_p},
      Neurologist == Non-Neurologist p-value = {eqa_hasAns__Neuro_Non_p}
      """)
print("")

#EQA_noAns - Epi vs Neuro, Epi vs. Non, Neuro vs Non
eqa_noAns__Epi_Neuro_stat, eqa_noAns__Epi_Neuro_p = stats.mannwhitneyu(
    eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1])
eqa_noAns__Epi_Non_stat, eqa_noAns__Epi_Non_p = stats.mannwhitneyu(
    eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 2])
eqa_noAns__Neuro_Non_stat, eqa_noAns__Neuro_Non_p = stats.mannwhitneyu(
    eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2])
print(f"""For EQA NoAns between Epileptologists, Neurologists, and Non-Neurologists,
      Epileptologist == Neurologist p-value = {eqa_noAns__Epi_Neuro_p},
      Epileptologist == Non-Neurologist p-value = {eqa_noAns__Epi_Non_p},
      Neurologist == Non-Neurologist p-value = {eqa_noAns__Neuro_Non_p}
      """)
print("")

fig, axs = plt.subplots(1, 1, dpi=600, figsize=(8,4))
sns.despine()

#plot boxplot
results = [eqa_hasAns_performance[:,0], eqa_hasAns_performance[:,1], eqa_hasAns_performance[:,2],
             eqa_noAns_performance[:,0], eqa_noAns_performance[:,1], eqa_noAns_performance[:,2]]
positions = [1, 1.5, 2,
             3.5, 4, 4.5,]
axs.boxplot(results, sym="", positions=positions)

#plot the individual points
#EQA hasAns
axs.plot(np.ones(5)*positions[0], eqa_hasAns_performance[:,0], '.', color=colors[0], alpha=0.75)
axs.plot(np.ones(5)*positions[1], eqa_hasAns_performance[:,1], '.', color=colors[1], alpha=0.75)
axs.plot(np.ones(5)*positions[2], eqa_hasAns_performance[:,2], '.', color=colors[2], alpha=0.75)
#EQA noAns
axs.plot(np.ones(5)*positions[3], eqa_noAns_performance[:,0], '.', color=colors[0], alpha=0.75)
axs.plot(np.ones(5)*positions[4], eqa_noAns_performance[:,1], '.', color=colors[1], alpha=0.75)
axs.plot(np.ones(5)*positions[5], eqa_noAns_performance[:,2], '.', color=colors[2], alpha=0.75)
axs.set_xticks(ticks=[1.5, 4])
axs.set_xticklabels(labels=['Extraction (Combined)\nWhen Outcome Exists', "Extraction (Combined)\nWhen Outcome Doesn't Exist"])

#hasAns significances
if eqa_hasAns__Epi_Neuro_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[0], positions[1]]), 
                     np.max([eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2]]) + lower_anno_raise],
                 xytext=[np.mean([positions[0], positions[1]]), 
                         np.max([eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=2.4, lengthB=0.25', lw=1))
if eqa_hasAns__Epi_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[0], positions[2]]), 
                     np.max([eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2]]) + upper_anno_raise],
                 xytext=[np.mean([positions[0], positions[2]]), 
                         np.max([eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2]]) + upper_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=4.8, lengthB=0.25', lw=1))
if eqa_hasAns__Neuro_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[1], positions[2]]), 
                     np.max([eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2]]) + lower_anno_raise],
                 xytext=[np.mean([positions[1], positions[2]]), 
                         np.max([eqa_hasAns_performance[:, 0], eqa_hasAns_performance[:, 1], eqa_hasAns_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=2.4, lengthB=0.25', lw=1))
    
    
    
#noAns significances
if eqa_noAns__Epi_Neuro_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[3], positions[4]]), 
                     np.max([eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2]]) + lower_anno_raise],
                 xytext=[np.mean([positions[3], positions[4]]), 
                         np.max([eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=2.4, lengthB=0.25', lw=1))
if eqa_noAns__Epi_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[3], positions[5]]), 
                     np.max([eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2]]) + upper_anno_raise],
                 xytext=[np.mean([positions[3], positions[5]]), 
                         np.max([eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2]]) + upper_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=4.8, lengthB=0.25', lw=1))
if eqa_noAns__Neuro_Non_p <= significance:
    axs.annotate("\u2217", 
                 xy=[np.mean([positions[4], positions[5]]), 
                     np.max([eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2]]) + lower_anno_raise],
                 xytext=[np.mean([positions[4], positions[5]]), 
                         np.max([eqa_noAns_performance[:, 0], eqa_noAns_performance[:, 1], eqa_noAns_performance[:, 2]]) + lower_anno_raise + 0.01],
                 bbox=dict(pad=0, fc='w', ec='none', boxstyle='square'),
                 ha='center',
                 va='center',
                 arrowprops=dict(arrowstyle='-[, widthB=2.4, lengthB=0.25', lw=1))
    

#https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
legend_elements = [Line2D([0], [0], color='k', marker='o', label='Epileptologist Notes', markerfacecolor=colors[0]),
                   Line2D([0], [0], color='k', marker='o', label='Neurologist Notes', markerfacecolor=colors[1]),
                   Line2D([0], [0], color='k', marker='o', label='Non-Neurologist Notes', markerfacecolor=colors[2])]

plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=False)
axs.set_ylabel('F$_1$ Score')
axs.set_ylim([0.35, 1.1])
axs.grid(True, alpha=0.25)
fig.savefig(f"{fig_save_dir}/fig_1b.png", dpi=600, bbox_inches='tight')
fig.savefig(f"{fig_save_dir}/fig_1b.pdf", dpi=600, bbox_inches='tight')