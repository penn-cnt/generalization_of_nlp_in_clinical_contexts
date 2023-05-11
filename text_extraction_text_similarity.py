import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import json
import Levenshtein
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
import statsmodels.api as sm
from scipy.stats import kruskal, mannwhitneyu, norm, sem
import annotation_utils as anno_utils
from rajpurkar_squad2_evaluate import compute_f1 #the squad2.0 evaluation code, available at https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

def get_eqa_annotation_text_label_and_contexts(proj_dir, subset=None):
    eqa_spans = {}
    
    #list all annotation directories. Inside each directory will be the tsv files for each annotator that has performed an annotation
    anno_docs = os.listdir(proj_dir)
    
    base_cols = ['Sen-Tok', 'Beg-End', 'Token']
    #collect annotations for each Document
    for doc in anno_docs:           
        
        #check if this doc should be included
        if subset != None:
            if doc not in subset:
                continue
        
        #find the document to be annotated and the annotators
        doc_dir = proj_dir+"/"+doc
        anno_files = os.listdir(doc_dir)

        #iterate through annotations for this document and get their annotation data
        annotations = {}
        for anno in anno_files:
            filename = doc_dir+"/"+anno
            anno_doc = pd.read_csv(filename, comment='#', sep='\t+',\
                         header=None, quotechar='"', engine='python', \
                         names=anno_utils.GetHeaders(filename), index_col=None)
            if len(anno_doc.columns) < 4:
                continue
            if 'HasSeizures' not in anno_doc.columns:
                anno_doc['HasSeizures'] = "_"
            if 'SeizureFrequency' not in anno_doc.columns:
                anno_doc['SeizureFrequency'] = "_"
            if 'TypeofSeizure' not in anno_doc.columns:
                anno_doc['TypeofSeizure'] = "_"
            if 'referenceRelation' in anno_doc.columns:
                anno_doc = anno_doc.drop('referenceRelation', axis=1)
            if 'referenceType' in anno_doc.columns:
                anno_doc = anno_doc.drop('referenceType', axis=1)
                
            #replace empty values and incomplete annotations
            anno_doc = anno_doc.replace(to_replace=r'\*.+|\*', value='_', regex=True)   
            anno_doc = anno_doc.fillna("_")
            
            #get the possible eqa labels
            possible_eqa_labels = set(anno_doc['SeizureFrequency'])
            
            #for each found label
            anno_context = None
            anno_pqf = None
            anno_elo = None
            for label in possible_eqa_labels:
                #if it's '_', it must be context
                if label == '_':
                    #find where the context is and get its text
                    context_doc = anno_doc.loc[anno_doc['SeizureFrequency'] == '_']
                    anno_context = anno_utils.series_to_paragraph(context_doc['Token'])
                #if it's a pqf, then overwrite the last pqf entry. IN THIS SCENARIO WE ONLY TAKE THE LAST PQF
                if 'Positive Quantitative Frequency' in label:
                    #find where the pqf is and get its text
                    pqf_doc = anno_doc.loc[anno_doc['SeizureFrequency'].str.contains('Positive Quantitative Frequency')]
                    anno_pqf = anno_utils.series_to_paragraph(pqf_doc['Token'])
                #if it's a elo, then overwrite the last elo entry. IN THIS SCENARIO WE ONLY TAKE THE LAST ELO
                if 'Explicit Last Occurrence' in label:
                    #find where the elo is and get its text
                    elo_doc = anno_doc.loc[anno_doc['SeizureFrequency'].str.contains('Explicit Last Occurrence')]
                    anno_elo = anno_utils.series_to_paragraph(elo_doc['Token'])
                    
            eqa_spans[doc] = {'context':anno_context, 'pqf_text':anno_pqf, 'elo_text':anno_elo}
    
    return eqa_spans

def combine_span_dicts(list_of_spans, label):
    #one array to hold the combined list, and the other to get the indices where each group ends and the other begins
    combined_list = []
    group_end_idx = [0]
    used_docs = []
    for spanlist in list_of_spans:
        combined_list.append([spanlist[doc_name][label] for doc_name in spanlist if not pd.isnull(spanlist[doc_name][label])])
        used_docs.append([doc_name for doc_name in spanlist if not pd.isnull(spanlist[doc_name][label])])
        group_end_idx.append(len(combined_list[-1])+group_end_idx[-1])
    #flatten the lists
    combined_list = sum(combined_list, [])
    used_docs = sum(used_docs, [])
    
    return combined_list, group_end_idx, used_docs

def remove_N_outliers(data_list, N):
    """removes the top and bottom N values from each dataset in list data_list"""
    processed_data = []
    for data in data_list:
        sorted_data = np.sort(data)
        processed_data.append(sorted_data[N:-N])
    return processed_data

def calculate_similarity_scores(proj_dir_epi, proj_dir_gen, proj_dir_neuro, subset_train=None, subset_test=None):
    #get the annotations   
    epi_train = get_eqa_annotation_text_label_and_contexts(proj_dir_epi, subset=subset_train)
    epi_test = get_eqa_annotation_text_label_and_contexts(proj_dir_epi, subset=subset_test)
    neuro_test = get_eqa_annotation_text_label_and_contexts(proj_dir_neuro)
    gen_test = get_eqa_annotation_text_label_and_contexts(proj_dir_gen)        
        
    print("Isolated annotations")
        
    #create a list that contains all phrases of yes, and one for all phrases of no, and one for idk if applicable
    all_contexts = combine_span_dicts([epi_train, epi_test, neuro_test, gen_test], 'context')
    all_pqfs = combine_span_dicts([epi_train, epi_test, neuro_test, gen_test], 'pqf_text')
    all_elos = combine_span_dicts([epi_train, epi_test, neuro_test, gen_test], 'elo_text')
    
    print("Combined text spans")
    
    #calculate all embeddings
    all_context_embeddings = np.array([sentence_mdl.encode(sentence) for sentence in all_contexts[0]])
    all_pqf_embeddings = np.array([sentence_mdl.encode(sentence) for sentence in all_pqfs[0]])
    all_elo_embeddings = np.array([sentence_mdl.encode(sentence) for sentence in all_elos[0]])   
    
    print("Calculated embeddings")
        
    #calculate levenshtein similarities pairwise across all documents
    pw_context_levenshtein = np.array([[Levenshtein.ratio(all_contexts[0][i], all_contexts[0][j]) for i in range(len(all_contexts[0]))] for j in range(len(all_contexts[0]))])
    pw_pqf_levenshtein = np.array([[Levenshtein.ratio(all_pqfs[0][i], all_pqfs[0][j]) for i in range(len(all_pqfs[0]))] for j in range(len(all_pqfs[0]))])
    pw_elo_levenshtein = np.array([[Levenshtein.ratio(all_elos[0][i], all_elos[0][j]) for i in range(len(all_elos[0]))] for j in range(len(all_elos[0]))])
    
    print("Calculated Levenshteins")
    
    #calculate cosine similarities pairwise across all documents
    pw_context_cos = np.array(util.cos_sim(all_context_embeddings, all_context_embeddings))
    pw_pqf_cos = np.array(util.cos_sim(all_pqf_embeddings, all_pqf_embeddings))
    pw_elo_cos = np.array(util.cos_sim(all_elo_embeddings, all_elo_embeddings))
    
    print("Calculated Cosines")
    
    return all_contexts, all_pqfs, all_elos, pw_context_levenshtein, pw_pqf_levenshtein, pw_elo_levenshtein, pw_context_cos, pw_pqf_cos, pw_elo_cos

def clean_mirrored_items(group_lev, group_cos):
    tril_idx = np.tril_indices(group_lev[-1].shape[0], k=0)
    group_lev[-1][tril_idx] = np.nan
    group_cos[-1][tril_idx] = np.nan
    return group_lev, group_cos

def flatten_and_drop_nans(group_lev, group_cos):
    group_lev[-1] = group_lev[-1].flatten()
    group_lev[-1] = group_lev[-1][~np.isnan(group_lev[-1])]
    group_cos[-1] = group_cos[-1].flatten()
    group_cos[-1] = group_cos[-1][~np.isnan(group_cos[-1])]
    return group_lev, group_cos
    
def get_grouped_comparisons(all_contexts, all_pqfs, all_elos, context_lev_sims, pqf_lev_sims, elo_lev_sims, context_cos_sims, pqf_cos_sims, elo_cos_sims, order = [(1,1), (1,2), (1,3), (1,4)], remove_outliers=5):
    #hold values in each grouped comparison, split by question
    group_pqf_lev = []
    group_elo_lev = []
    group_pqf_cos = []
    group_elo_cos = []
    
    #hold values in each grouped comparison, regardless of question
    group_all_lev = []
    group_all_cos = []
    group_context_lev = []
    group_context_cos = []
    
    #keep track of the group comparison labels
    labels=[]
    
    #create groups
    for i,j in order:
        
        #slice into the 2D matrix for contexts
        group_context_lev.append(context_lev_sims[all_contexts[1][i-1]:all_contexts[1][i], all_contexts[1][j-1]:all_contexts[1][j]])
        group_context_cos.append(context_cos_sims[all_contexts[1][i-1]:all_contexts[1][i], all_contexts[1][j-1]:all_contexts[1][j]])
        
        #for pqfs
        group_pqf_lev.append(pqf_lev_sims[all_pqfs[1][i-1]:all_pqfs[1][i], all_pqfs[1][j-1]:all_pqfs[1][j]])
        group_pqf_cos.append(pqf_cos_sims[all_pqfs[1][i-1]:all_pqfs[1][i], all_pqfs[1][j-1]:all_pqfs[1][j]])
        
        #for elos
        group_elo_lev.append(elo_lev_sims[all_elos[1][i-1]:all_elos[1][i], all_elos[1][j-1]:all_elos[1][j]])
        group_elo_cos.append(elo_cos_sims[all_elos[1][i-1]:all_elos[1][i], all_elos[1][j-1]:all_elos[1][j]])
        
        labels.append(f"{groups[i-1]} - {groups[j-1]}")

        #if we're on the main diagonal, then we're going to have mirrored items. Remove the duplicates. #we also don't care about the main diagonal itself, so we can remove those too
        if i == j:
            group_context_lev, group_context_cos = clean_mirrored_items(group_context_lev, group_context_cos)
            group_pqf_lev, group_pqf_cos = clean_mirrored_items(group_pqf_lev, group_pqf_cos)
            group_elo_lev, group_elo_cos = clean_mirrored_items(group_elo_lev, group_elo_cos)

        #now, flatten the arrays and drop nans
        group_context_lev, group_context_cos = flatten_and_drop_nans(group_context_lev, group_context_cos)
        group_pqf_lev, group_pqf_cos = flatten_and_drop_nans(group_pqf_lev, group_pqf_cos)
        group_elo_lev, group_elo_cos = flatten_and_drop_nans(group_elo_lev, group_elo_cos)

        #append values into the containers that hold all similarities
        group_all_lev.append(np.concatenate([group_pqf_lev[-1], group_elo_lev[-1]]))
        group_all_cos.append(np.concatenate([group_pqf_cos[-1], group_elo_cos[-1]]))

    #remove outliers in each question group
    group_context_lev = remove_N_outliers(group_context_lev, remove_outliers)
    group_context_cos = remove_N_outliers(group_context_cos, remove_outliers)
    group_pqf_lev = remove_N_outliers(group_pqf_lev, remove_outliers)
    group_pqf_cos = remove_N_outliers(group_pqf_cos, remove_outliers)
    group_elo_lev = remove_N_outliers(group_elo_lev, remove_outliers)
    group_elo_cos = remove_N_outliers(group_elo_cos, remove_outliers)
        
    #remove outliers in combined groups
    group_all_lev = remove_N_outliers(group_all_lev, remove_outliers)
    group_all_cos = remove_N_outliers(group_all_cos, remove_outliers)
    
    return group_context_lev, group_pqf_lev, group_elo_lev, group_context_cos, group_pqf_cos, group_elo_cos, group_all_lev, group_all_cos, labels

def load_json_dataset(file_path):
    #open the json as a list of datum
    with open(file_path, 'r') as f:
        dataset = json.load(f)['data']
    #iterate through the list. For each one, add it to the final dataset dictionary
    return {datum['id']:{'answers':datum['answers'], 'question':datum['question'], 'context':datum['context']} for datum in dataset}

def sim_to_train(doc_id):
    """
    Calculates the levenshtein and cosine similarity for a specific doc_id from the all_model_preds table.
    This function uses variables outside the scope of the function.
    doc_id: The doc's ID, as the index of the table
    """
    #prepare to find which index in the similarity matrices to use
    #the doc_id has an extra _pqf or _elo at the end of it. We can remove it by splitting and taking everything but the last one
    doc_name = "_".join(doc_id.split("_")[:-1])

    #find where doc_name occurs in all_contexts, and (all_pqfs, or all_elos)
    #the doc_names have a possible #.txt at the end of them, precluding simple matching. We need a list comprehension to do so
    if 'pqf' in doc_id:
        doc_idx = [idx for idx,x in enumerate(all_pqfs[2]) if doc_name in x]
    else:
        doc_idx = [idx for idx,x in enumerate(all_elos[2]) if doc_name in x]
    doc_idx_context = [idx for idx,x in enumerate(all_contexts[2]) if doc_name in x]
    
    #find the average similarities this doc has from the epi_train notes
    if 'pqf' in doc_id:
        avg_lev = np.mean(pw_pqf_levenshtein[all_pqfs[1][0]:all_pqfs[1][1], doc_idx])
        avg_cos = np.mean(pw_pqf_cos[all_pqfs[1][0]:all_pqfs[1][1], doc_idx])
    else:
        avg_lev = np.mean(pw_elo_levenshtein[all_elos[1][0]:all_elos[1][1], doc_idx])
        avg_cos = np.mean(pw_elo_cos[all_elos[1][0]:all_elos[1][1], doc_idx])
    
    avg_lev_context = np.mean(pw_context_levenshtein[all_contexts[1][0]:all_contexts[1][1], doc_idx_context])
    avg_cos_context = np.mean(pw_context_cos[all_contexts[1][0]:all_contexts[1][1], doc_idx_context])
        
    return pd.Series({"avg_lev":avg_lev, "avg_cos":avg_cos, "avg_lev_context":avg_lev_context, "avg_cos_context":avg_cos_context})

#where to save the figures
fig_save_dir = 'figures'

#where are the annotations stored?
proj_dir_epi = 'epipleptologist_annotations/' 
proj_dir_neuro = 'neurologist_annotations/'
proj_dir_general = 'non_epileptologist_annotations/'

#where is the generalization test set and the JAMIA training set stored, so we can properly compare apples to apples?
#we load the files from the classification task, but we only realy need the datum ID's to get their annotations, which is what we care about later on.
epi_test_set_path = 'penn_generalization_eval_datasets/classification_epi.json'
jamia_train_set_path = 'JAMIA/hasSz_train_1.0.json'

#open the jamia train set and get the annotation filenames that were used
with open(jamia_train_set_path, 'r') as f:
    jamia_train_files = [json.loads(datum)['title'] for datum in f.read().splitlines()]

#open the epi generalization test set and get the annotation filenames that were used
with open(epi_test_set_path, 'r') as f:
    epi_test_files = [f"{'_'.join(json.loads(datum)['id'].split('_')[:-1])}.txt" for datum in f.read().splitlines()]
    
#where are the transformer models stored?
sentence_mdl_path = 'huggingface_models/all-MiniLM-L6-v2/'
sentence_mdl = SentenceTransformer(sentence_mdl_path)

#calculate similarity scores
all_contexts, all_pqfs, all_elos, pw_context_levenshtein, pw_pqf_levenshtein, pw_elo_levenshtein, pw_context_cos, pw_pqf_cos, pw_elo_cos = \
    calculate_similarity_scores(proj_dir_epi, proj_dir_gen, proj_dir_neuro, subset_train=jamia_train_files, subset_test=epi_test_files)
    
#create comparison groups
group_context_lev, group_pqf_lev, group_elo_lev, group_context_cos, group_pqf_cos, group_elo_cos, group_all_lev, group_all_cos, group_labels = \
    get_grouped_comparisons(all_contexts, all_pqfs, all_elos, \
                            pw_context_levenshtein, pw_pqf_levenshtein, pw_elo_levenshtein, \
                            pw_context_cos, pw_pqf_cos, pw_elo_cos)

#load eqa datasets so we can access the ground truths
epi_train_dataset = load_json_dataset(f'penn_generalization_eval_datasets/train_eqa_epi.json')
epi_test_dataset = load_json_dataset(f'penn_generalization_eval_datasets/eqa_epi.json')
neuro_test_dataset = load_json_dataset(f'penn_generalization_eval_datasets/eqa_neuro.json')
gen_test_dataset = load_json_dataset(f'penn_generalization_eval_datasets/eqa_general.json')


#iterate through the model predictions. Then, load the model prediction tsv files and get their predictions for each document
#we want to create a table of Doc_ID | Prov Type | Ground truth | Pred seed 2| Pred seed 17 | ... | Pred seed 136
    #we start with a dict of format {DOC_ID: {Prov Type:..., Ground Truth:..., ...}, ...}
all_model_preds = {}
for pred_dir in os.listdir('model_generalization_predictions'):
    
    #skip everything except hasSz predictions
    if 'szFreq' not in pred_dir:
        continue
    
    #load the predictions
    with open(f"model_generalization_predictions/{pred_dir}/eval_predictions.json", 'r') as f:
        szFreq_pred = json.load(f)
    
    #iterate through the predictions
    for pred_id in szFreq_pred:
            
        #the directory name has some model info in it
        model_info = pred_dir.split("_")
        
        #get the ground truth prediction for this one
        gt = None
        if model_info[1] == 'general':
            gt = gen_test_dataset[pred_id]['answers']['text']
        elif model_info[1] == 'neuro':
            gt = neuro_test_dataset[pred_id]['answers']['text']
        elif model_info[1] == 'epi':
            gt = epi_test_dataset[pred_id]['answers']['text']
        else:
            raise
        
        #check if this doc has already been added to the dict. create an entry if not
        if pred_id not in all_model_preds:
            #process the 
            all_model_preds[pred_id] = {
                'prov_type':model_info[1], 
                'ground_truth':gt,
            }
        
        #update the entry
        #first, check if the answer is empty
        if all_model_preds[pred_id]['ground_truth'] == []:
            all_model_preds[pred_id][f"seed_{model_info[-1]}_f1"] = compute_f1("", szFreq_pred[pred_id])
        else: #otherwise, get the highest f1 score
            all_model_preds[pred_id][f"seed_{model_info[-1]}_f1"] = np.max([compute_f1(ans, szFreq_pred[pred_id]) for ans in all_model_preds[pred_id]['ground_truth']])       
        
            
#convert to pandas DF
all_model_preds = pd.DataFrame(all_model_preds).transpose()
#drop all no-answer rows, as these lack kernels
all_model_preds = all_model_preds.loc[all_model_preds['ground_truth'].str.len() != 0]

#calculate the overall F1 of each document.
all_model_preds['overall_F1'] = all_model_preds.apply(lambda x: np.mean([x.seed_2_f1, x.seed_17_f1, x.seed_136_f1, x.seed_42_f1, x.seed_97_f1]), axis=1)

#for each document, calculate its average levenshtein and cosine similarity to the epileptologist training documents
all_model_preds = all_model_preds.join(pd.DataFrame(all_model_preds.apply(lambda x: sim_to_train(x.name), axis=1)))

#Perform Mann Whitney U tests between distributions of similarities - are the distributions significantly different in correct vs incorrect findings?
min_correct_threshold = 0.5
#isolate the correct and incorrect preds. We ignore idk ground truths
correct_preds = all_model_preds.loc[all_model_preds.overall_F1 >= min_correct_threshold]
incorrect_preds = all_model_preds.loc[all_model_preds.overall_F1 < min_correct_threshold]

print(f"There are {len(correct_preds)} samples with F1 >= {min_correct_threshold}, and {len(incorrect_preds)} with F1 < {min_correct_threshold}")
print(f"2S MWU Test for avg_lev: {mannwhitneyu(correct_preds.avg_lev, incorrect_preds.avg_lev)}")
print(f"2S MWU Test for avg_cos: {mannwhitneyu(correct_preds.avg_cos, incorrect_preds.avg_cos)}")
print(f"2S MWU Test for avg_lev_context: {mannwhitneyu(correct_preds.avg_lev_context, incorrect_preds.avg_lev_context)}")
print(f"2S MWU Test for avg_cos_context: {mannwhitneyu(correct_preds.avg_cos_context, incorrect_preds.avg_cos_context)}")

#plot the distributions as box plots. 
positions_correct=[1, 3]
positions_incorrect=[1.5, 3.5]
colors=['#0075DC', '#FFC107']
labels=['Kernel', 'Context']
fig, axs = plt.subplots(1, 2, figsize=(14,5))
sns.despine()
#levenshtein similarities
bp1=axs[0].boxplot([correct_preds.avg_lev, correct_preds.avg_lev_context], 
               sym="", positions=positions_correct, patch_artist=True, notch=True, 
               boxprops={'facecolor':colors[0], 'color':colors[0]}, capprops={'color':colors[0]}, 
               whiskerprops={'color':colors[0]}, flierprops={'color':colors[0], 'markeredgecolor':colors[0]}, 
               medianprops={'color':colors[0]}, widths=0.4)
bp2=axs[0].boxplot([incorrect_preds.avg_lev, incorrect_preds.avg_lev_context], 
               sym="", positions=positions_incorrect, patch_artist=True, notch=True, 
               boxprops={'facecolor':colors[1], 'color':colors[1]}, capprops={'color':colors[1]}, 
               whiskerprops={'color':colors[1]}, flierprops={'color':colors[1], 'markeredgecolor':colors[1]}, 
               medianprops={'color':colors[1]}, widths=0.4)
axs[0].set_ylabel('Mean Levenshtein\nSimilarity to Training Documents')
axs[0].set_xticks([1.25, 3.25])
axs[0].set_xticklabels(labels, horizontalalignment='center')
axs[0].legend([bp1['boxes'][0], bp2['boxes'][0]], [f'F$_1$ $\geq$ {min_correct_threshold}', f'F$_1$  < {min_correct_threshold}'])
axs[0].set_ylim([0.2, 0.63])
#cosine similarities
axs[1].boxplot([correct_preds.avg_cos, correct_preds.avg_cos_context], 
               sym="", positions=positions_correct, patch_artist=True, notch=True, 
               boxprops={'facecolor':colors[0], 'color':colors[0]}, capprops={'color':colors[0]}, 
               whiskerprops={'color':colors[0]}, flierprops={'color':colors[0], 'markeredgecolor':colors[0]}, 
               medianprops={'color':colors[0]}, widths=0.4)
axs[1].boxplot([incorrect_preds.avg_cos, incorrect_preds.avg_cos_context], 
               sym="", positions=positions_incorrect, patch_artist=True, notch=True, 
               boxprops={'facecolor':colors[1], 'color':colors[1]}, capprops={'color':colors[1]}, 
               whiskerprops={'color':colors[1]}, flierprops={'color':colors[1], 'markeredgecolor':colors[1]}, 
               medianprops={'color':colors[1]}, widths=0.4)
axs[1].set_ylabel('Mean Cosine\nSimilarity to Training Documents')
axs[1].set_xticks([1.25, 3.25])
axs[1].set_xticklabels(labels, horizontalalignment='center')
axs[1].set_ylim([0.05, 0.85])
plt.savefig(f"{fig_save_dir}/fig_2b.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{fig_save_dir}/fig_2b.pdf", dpi=600, bbox_inches='tight')
plt.show()


#Create QQ plots of overall distributions, including those from the classification task
#load the hasSz similarity data
def combine_similarity_lists(l1, l2):
    if len(l1) != len(l2):
        print("Err: list lengths unequal")
        return None
    combined_list = []
    for i in range(len(l1)):
        combined_list.append(np.concatenate((l1[i], l2[i])))
    return combined_list
with open('hasSz_similarities.pkl', 'rb') as f:
    hasSz_similarities = pickle.load(f)
    lev_hasSz = hasSz_similarities['lev_hasSz']
    cos_hasSz = hasSz_similarities['cos_hasSz']
    lev_context_hasSz = hasSz_similarities['lev_context_hasSz']
    cos_context_hasSz = hasSz_similarities['cos_context_hasSz']
    model_preds_hasSz = hasSz_similarities['model_preds_hasSz']
    
#combine hasSz and EQA kernels and contexts
lev_kernel = pd.DataFrame(combine_similarity_lists(lev_hasSz, group_all_lev)).transpose()
cos_kernel = pd.DataFrame(combine_similarity_lists(cos_hasSz, group_all_cos)).transpose()
lev_context = pd.DataFrame(combine_similarity_lists(lev_context_hasSz, group_context_lev)).transpose()
cos_context = pd.DataFrame(combine_similarity_lists(cos_context_hasSz, group_context_cos)).transpose()

#calculate percentiles for use in Q-Q plots
num_percentiles=100
lev_quantiles = [np.nanpercentile(lev_kernel[i], np.linspace(0, 100, num_percentiles)) for i in lev_kernel.columns]
cos_quantiles = [np.nanpercentile(cos_kernel[i], np.linspace(0, 100, num_percentiles)) for i in cos_kernel.columns]
lev_context_quantiles = [np.nanpercentile(lev_context[i], np.linspace(0, 100, num_percentiles)) for i in lev_context.columns]
cos_context_quantiles = [np.nanpercentile(cos_context[i], np.linspace(0, 100, num_percentiles)) for i in cos_context.columns]
lev_medians = lev_kernel.apply(np.nanmedian)
cos_medians = cos_kernel.apply(np.nanmedian)
lev_context_medians = lev_context.apply(np.nanmedian)
cos_context_medians = cos_context.apply(np.nanmedian)

#create Q-Q plots comparing distributions
fig, axs = plt.subplots(2, 2, figsize=(16,16))
sns.despine()
colors = ['#B50A49', '#1E88E5', '#FFC107', '#004D40']

#levenshtein kernel similarities
for i in range(1, len(lev_kernel.columns)):
    axs[0,0].scatter(lev_quantiles[0], lev_quantiles[i], color=colors[i], alpha=0.66)
axs[0,0].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k')
for i in range(1, len(lev_kernel.columns)):
    axs[0,0].hlines(lev_medians[i], 0, lev_medians[0], colors=colors[i], linestyle='dashed', alpha=0.66)
axs[0,0].vlines(lev_medians[0], 0, np.max(lev_medians[1:]), colors=colors[0], linestyle='dashed', alpha=0.66)
axs[0,0].legend(['Epileptologist Training vs. Epileptologist Validation', 'Epileptologist Training vs. Neurologist', 'Epileptologist Training vs. Non-Neurologist'])
axs[0,0].set_xlim([0,1.05]);axs[0,0].set_ylim([0,1.05])

#cosine kernel similarities
for i in range(1, len(cos_kernel.columns)):
    axs[0,1].scatter(cos_quantiles[0], cos_quantiles[i], color=colors[i], alpha=0.66)
axs[0,1].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k')
for i in range(1, len(cos_kernel.columns)):
    axs[0,1].hlines(cos_medians[i], 0, cos_medians[0], colors=colors[i], linestyle='dashed', alpha=0.66)
axs[0,1].vlines(cos_medians[0], 0, np.max(cos_medians[1:]), colors=colors[0], linestyle='dashed', alpha=0.66)
axs[0,1].set_xlim([0,1.05]);axs[0,1].set_ylim([0,1.05])

#levenshtein context similarities
for i in range(1, len(lev_context.columns)):
    axs[1,0].scatter(lev_context_quantiles[0], lev_context_quantiles[i], color=colors[i], alpha=0.66)
axs[1,0].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k')
for i in range(1, len(lev_context.columns)):
    axs[1,0].hlines(lev_context_medians[i], 0, lev_context_medians[0], colors=colors[i], linestyle='dashed', alpha=0.66)
axs[1,0].vlines(lev_context_medians[0], 0, np.max(lev_context_medians[1:]), colors=colors[0], linestyle='dashed', alpha=0.66)
axs[1,0].set_xlim([0,1.05]);axs[1,0].set_ylim([0,1.05])

#cosine context similarities
for i in range(1, len(cos_context.columns)):
    axs[1,1].scatter(cos_context_quantiles[0], cos_context_quantiles[i], color=colors[i], alpha=0.66)
axs[1,1].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k')
for i in range(1, len(cos_context.columns)):
    axs[1,1].hlines(cos_context_medians[i], 0, cos_context_medians[0], colors=colors[i], linestyle='dashed', alpha=0.66)
axs[1,1].vlines(cos_context_medians[0], 0, np.max(cos_context_medians[1:]), colors=colors[0], linestyle='dashed', alpha=0.66)
axs[1,1].set_xlim([0,1.05]);axs[1,1].set_ylim([0,1.05])

#label axes top row
axs[0,0].set_ylabel('Levenshtein Similarity\n(Epileptologist Training Set vs. Testing Sets)')
axs[0,1].set_ylabel('Cosine Similarity\n(Epileptologist Training Set vs. Testing Sets')
axs[0,0].set_xlabel('Levenshtein Similarity\n(Epileptologist Training Set Self-Similarity)')
axs[0,1].set_xlabel('Cosine Similarity\n(Epileptologist Training Set Self-Similarity)')
#label axes bottom row
axs[1,0].set_ylabel('Levenshtein Similarity\n(Epileptologist Training Set vs. Testing Sets)')
axs[1,1].set_ylabel('Cosine Similarity\n(Epileptologist Training Set vs. Testing Sets')
axs[1,0].set_xlabel('Levenshtein Similarity\n(Epileptologist Training Set Self-Similarity)')
axs[1,1].set_xlabel('Cosine Similarity\n(Epileptologist Training Set Self-Similarity)')
plt.savefig(f"{fig_save_dir}/fig_3.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{fig_save_dir}/fig_3.pdf", dpi=600, bbox_inches='tight')
plt.show()