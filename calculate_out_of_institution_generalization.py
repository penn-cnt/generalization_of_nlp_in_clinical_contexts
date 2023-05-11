import pandas as pd
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
from utils import load_boolq_formatted_dataset, load_squad_formatted_dataset
import seaborn as sns
from rajpurkar_squad2_evaluate import compute_f1 #the squad2.0 evaluation code, available at https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/



#where to save the figures
fig_save_dir = r'figures/'

#epileptologist dataset paths
epi_hasSz_dataset_path = 'penn_generalization_eval_datasets/classification_epi.json'
epi_szFreq_dataset_path = 'penn_generalization_eval_datasets/eqa_epi.json'
#neurologist dataset paths
neuro_hasSz_dataset_path = 'penn_generalization_eval_datasets/classification_neuro.json'
neuro_szFreq_dataset_path = 'penn_generalization_eval_datasets/eqa_neuro.json'
#non-neurologist dataset directory
general_hasSz_dataset_path = 'penn_generalization_eval_datasets/classification_general.json'
general_szFreq_dataset_path = 'penn_generalization_eval_datasets/eqa_general.json'

#epileptologist prediction paths
epi_hasSz_prediction_path = r'model_generalization_predictions/hasSz_epi_NOTES_MODEL_17/eval_predictions.tsv'
epi_szFreq_prediction_path = r'model_generalization_predictions/szFreq_epi_NOTES_MODEL_17/eval_predictions.json'
#neurologist prediction paths
neuro_hasSz_prediction_path = r'model_generalization_predictions/hasSz_neuro_NOTES_MODEL_17/eval_predictions.tsv'
neuro_szFreq_prediction_path = r'model_generalization_predictions/szFreq_neuro_NOTES_MODEL_17/eval_predictions.json'
#non-neurologist prediction paths
general_hasSz_prediction_path = r'model_generalization_predictions/hasSz_general_NOTES_MODEL_17/eval_predictions.tsv'
general_szFreq_prediction_path = r'model_generalization_predictions/szFreq_general_NOTES_MODEL_17/eval_predictions.json'

#load the datasets
epi_hasSz_dataset = load_boolq_formatted_dataset(epi_hasSz_dataset_path)
epi_szFreq_dataset = load_squad_formatted_dataset(epi_szFreq_dataset_path)['data']
neuro_hasSz_dataset = load_boolq_formatted_dataset(neuro_hasSz_dataset_path)
neuro_szFreq_dataset = load_squad_formatted_dataset(neuro_szFreq_dataset_path)['data']
general_hasSz_dataset = load_boolq_formatted_dataset(general_hasSz_dataset_path)
general_szFreq_dataset = load_squad_formatted_dataset(general_szFreq_dataset_path)['data']

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
    
#merge predictions and dataset text
one_hot = {'Yes':1, 'No':0, 'no-answer': 2}
all_predictions = {}

#epileptologists.
for i in range(len(epi_hasSz_dataset)):
    #get the doc_id of the datum
    doc_id = "_".join(epi_hasSz_dataset[i]['id'].split("_")[:-1])
    if doc_id in all_predictions: #there should be no overlap between the datasets
        print("WARNING: THIS EPI DOCUMENT IS NOT UNIQUE")
        
    #load basic info of the passage
    all_predictions[doc_id] = {}
    all_predictions[doc_id]['context'] = epi_hasSz_dataset[i]['passage']
    all_predictions[doc_id]['type'] = 'epileptologist'
    
    #get the answers for hasSz
    all_predictions[doc_id]['hasSz_gt'] = epi_hasSz_dataset[i]['label']
    
    #get the prediction for hasSz
    #verify that the ID is correct
    if "_".join(epi_hasSz_preds.iloc[i]['ID'].split("_")[:-1]) == doc_id:
        all_predictions[doc_id]['hasSz_pred'] = epi_hasSz_preds.iloc[i]['argmax']
    else:
        print("WARNING, PREDICTION - DATASET INDEXING DESYNCHRONIZED | HasSz Data - HasSz Pred")
    
    #get the answers and predictions for pqf/elo
    for j in range(2):
        #check that the hasSz and SzFreq datasets line up
        if "_".join(epi_szFreq_dataset[2*i + j]['id'].split("_")[:-1]) != doc_id:
            print("WARNING, DATASET - DATASET INDEXING DESYNCHONIZED | SzFreq - HasSz")
        if 'pqf' in epi_szFreq_dataset[2*i + j]['id']:
            all_predictions[doc_id]['pqf_gt'] = epi_szFreq_dataset[2*i + j]['answers']['text']
            all_predictions[doc_id]['pqf_pred'] = epi_szFreq_preds[epi_szFreq_dataset[2*i + j]['id']]
        else:
            all_predictions[doc_id]['elo_gt'] = epi_szFreq_dataset[2*i + j]['answers']['text']
            all_predictions[doc_id]['elo_pred'] = epi_szFreq_preds[epi_szFreq_dataset[2*i + j]['id']]
            
#neurologists.
for i in range(len(neuro_hasSz_dataset)):
    #get the doc_id of the datum
    doc_id = "_".join(neuro_hasSz_dataset[i]['id'].split("_")[:-1])
    if doc_id in all_predictions: #there should be no overlap between the datasets
        print("WARNING: THIS EPI DOCUMENT IS NOT UNIQUE")
        
    #load basic info of the passage
    all_predictions[doc_id] = {}
    all_predictions[doc_id]['context'] = neuro_hasSz_dataset[i]['passage']
    all_predictions[doc_id]['type'] = 'neurologist'
    
    #get the answers for hasSz
    all_predictions[doc_id]['hasSz_gt'] = neuro_hasSz_dataset[i]['label']
    
    #get the prediction for hasSz
    #verify that the ID is correct
    if "_".join(neuro_hasSz_preds.iloc[i]['ID'].split("_")[:-1]) == doc_id:
        all_predictions[doc_id]['hasSz_pred'] = neuro_hasSz_preds.iloc[i]['argmax']
    else:
        print("WARNING, PREDICTION - DATASET INDEXING DESYNCHRONIZED | HasSz Data - HasSz Pred")
    
    #get the answers and predictions for pqf/elo
    for j in range(2):
        #check that the hasSz and SzFreq datasets line up
        if "_".join(neuro_szFreq_dataset[2*i + j]['id'].split("_")[:-1]) != doc_id:
            print("WARNING, DATASET - DATASET INDEXING DESYNCHONIZED | SzFreq - HasSz")
        if 'pqf' in neuro_szFreq_dataset[2*i + j]['id']:
            all_predictions[doc_id]['pqf_gt'] = neuro_szFreq_dataset[2*i + j]['answers']['text']
            all_predictions[doc_id]['pqf_pred'] = neuro_szFreq_preds[neuro_szFreq_dataset[2*i + j]['id']]
        else:
            all_predictions[doc_id]['elo_gt'] = neuro_szFreq_dataset[2*i + j]['answers']['text']
            all_predictions[doc_id]['elo_pred'] = neuro_szFreq_preds[neuro_szFreq_dataset[2*i + j]['id']]

#non-neurologists.
for i in range(len(general_hasSz_dataset)):
    #get the doc_id of the datum
    doc_id = "_".join(general_hasSz_dataset[i]['id'].split("_")[:-1])
    if doc_id in all_predictions: #there should be no overlap between the datasets
        print("WARNING: THIS EPI DOCUMENT IS NOT UNIQUE")
        
    #load basic info of the passage
    all_predictions[doc_id] = {}
    all_predictions[doc_id]['context'] = general_hasSz_dataset[i]['passage']
    all_predictions[doc_id]['type'] = 'generalist'
    
    #get the answers for hasSz
    all_predictions[doc_id]['hasSz_gt'] = general_hasSz_dataset[i]['label']
    
    #get the prediction for hasSz
    #verify that the ID is correct
    if "_".join(general_hasSz_preds.iloc[i]['ID'].split("_")[:-1]) == doc_id:
        all_predictions[doc_id]['hasSz_pred'] = general_hasSz_preds.iloc[i]['argmax']
    else:
        print("WARNING, PREDICTION - DATASET INDEXING DESYNCHRONIZED | HasSz Data - HasSz Pred")
    
    #get the answers and predictions for pqf/elo
    for j in range(2):
        #check that the hasSz and SzFreq datasets line up
        if "_".join(general_szFreq_dataset[2*i + j]['id'].split("_")[:-1]) != doc_id:
            print("WARNING, DATASET - DATASET INDEXING DESYNCHONIZED | SzFreq - HasSz")
        if 'pqf' in general_szFreq_dataset[2*i + j]['id']:
            all_predictions[doc_id]['pqf_gt'] = general_szFreq_dataset[2*i + j]['answers']['text']
            all_predictions[doc_id]['pqf_pred'] = general_szFreq_preds[general_szFreq_dataset[2*i + j]['id']]
        else:
            all_predictions[doc_id]['elo_gt'] = general_szFreq_dataset[2*i + j]['answers']['text']
            all_predictions[doc_id]['elo_pred'] = general_szFreq_preds[general_szFreq_dataset[2*i + j]['id']]
            
#convert to a dataframe
all_predictions = pd.DataFrame(all_predictions).transpose()

#save the predictions for manual evaluation
# all_predictions.to_excel("prediction_correctness.xlsx")

#open the manually evaluated results
prediction_correctness = pd.read_excel("prediction_correctness_complete.xlsx")
epileptologist_pqf_agg = np.nanmean(prediction_correctness.loc[prediction_correctness['type'] == 'epileptologist']['pqf_correct'])
epileptologist_elo_agg = np.nanmean(prediction_correctness.loc[prediction_correctness['type'] == 'epileptologist']['elo_correct'])
neurologist_pqf_agg = np.nanmean(prediction_correctness.loc[prediction_correctness['type'] == 'neurologist']['pqf_correct'])
neurologist_elo_agg = np.nanmean(prediction_correctness.loc[prediction_correctness['type'] == 'neurologist']['elo_correct'])
generalist_pqf_agg = np.nanmean(prediction_correctness.loc[prediction_correctness['type'] == 'generalist']['pqf_correct'])
generalist_elo_agg = np.nanmean(prediction_correctness.loc[prediction_correctness['type'] == 'generalist']['elo_correct'])

#from Author S.W.T.'s manual adjudication
michigan_hasSz_agg = 0.74
michigan_pqf_correct = 0.89
michigan_elo_correct = 0.87 

print(f"epileptologist hasSz agreement: {epileptologist_hasSz_agg}")
print(f"neurologist hasSz agreement: {neurologist_hasSz_agg}")
print(f"generalist hasSz agreement: {generalist_hasSz_agg}")
print(f"epileptologist pqf agreement: {epileptologist_pqf_agg}")
print(f"neurologist pqf agreement: {neurologist_pqf_agg}")
print(f"generalist pqf agreement: {generalist_pqf_agg}")
print(f"epileptologist elo agreement: {epileptologist_elo_agg}")
print(f"neurologist elo agreement: {neurologist_elo_agg}")
print(f"generalist elo agreement: {generalist_elo_agg}")

#plot results
x = np.arange(0, 12).astype(float)
colors = ['#B50A49', '#1E88E5', '#FFC107', '#004D40']
fontsize=12
divider = 1.5
width=1
x[4:] += divider
x[8:] += divider
y = [epileptologist_hasSz_agg, neurologist_hasSz_agg, generalist_hasSz_agg, michigan_hasSz_agg, epileptologist_pqf_agg, neurologist_pqf_agg, generalist_pqf_agg, 
     michigan_pqf_correct, epileptologist_elo_agg, neurologist_elo_agg, generalist_elo_agg, michigan_elo_correct]

plt.figure(figsize=(10,6))
sns.despine()
plt.bar(x[0],   y[0], width=width, color=colors[0], edgecolor='black')
plt.bar(x[1],   y[1], width=width, color=colors[1], edgecolor='black')
plt.bar(x[2],   y[2], width=width, color=colors[2], edgecolor='black')
plt.bar(x[3],   y[3], width=width, color=colors[3], edgecolor='black')

plt.bar(x[4],   y[4], width=width, color=colors[0], edgecolor='black')
plt.bar(x[5],   y[5], width=width, color=colors[1], edgecolor='black')
plt.bar(x[6],   y[6], width=width, color=colors[2], edgecolor='black')
plt.bar(x[7],   y[7], width=width, color=colors[3], edgecolor='black')

plt.bar(x[8],   y[8], width=width, color=colors[0], edgecolor='black')
plt.bar(x[9],   y[9], width=width, color=colors[1], edgecolor='black')
plt.bar(x[10],   y[10], width=width, color=colors[2], edgecolor='black')
plt.bar(x[11],   y[11], width=width, color=colors[3], edgecolor='black')
plt.legend(['Epileptologist Notes', 'Neurologist Notes', 'Non-Neurologist Notes', 'Out-Of-Institution Notes'], loc='lower right', fontsize=fontsize)
plt.xticks(ticks=[np.mean(x[1:3]), np.mean(x[5:7]), np.mean(x[9:11])],
           labels=['Seizure Freedom\nClassification', 'Seizure Frequency\nExtraction', 'Date of Last Seizure\nExtraction'],
           rotation = 0, ha='center', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel("Agreement", fontsize=fontsize)
plt.savefig(f"{fig_save_dir}/fig_4.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{fig_save_dir}/fig_4.pdf", dpi=600, bbox_inches='tight')
plt.show()