import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import pickle
import Levenshtein
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
import statsmodels.api as sm
from scipy.stats import kruskal, mannwhitneyu, norm, sem
import annotation_utils as anno_utils

def exclude_hasSz_annotation_text_and_label(proj_dir, subset=None):
    """
    Get the Surrounding Context of each paragraph
    """
    hasSz_spans = {}
    
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
        
            #find where the annotations for HasSz are not
            hasSz_anno_exc = anno_doc.loc[(anno_doc['HasSeizures'] == '_')]
            hasSz_anno = set(anno_doc['HasSeizures'])
            hasSz_anno.remove("_")
            
            #if there are multiple statements of hasSz or no statements, then print a warning
            if len(hasSz_anno) == 1:
                label="".join([ch for ch in hasSz_anno.pop() if ch.isalpha()])
                
                #if the label is Unspecified, then include the entire paragraph, as the missing part is just 'HPI' or 'History' or something
                if label=='Unspecified':
                    hasSz_spans[doc] = {'text':anno_utils.series_to_paragraph(anno_doc['Token']), 'label':label}
                else:
                    hasSz_spans[doc] = {'text':anno_utils.series_to_paragraph(hasSz_anno_exc['Token']), 'label':label}
            else:
                print(f"WARNING: Improper HasSz annotation detected at {doc}")
                hasSz_spans[doc] = hasSz_anno_exc
                
    return hasSz_spans

def get_hasSz_annotation_text_and_label(proj_dir, subset=None):
    """
    Get the Kernel of each paragraph
    """
    hasSz_spans = {}
    
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
        
            #find where the annotations for HasSz are
            hasSz_anno_doc = anno_doc.loc[(anno_doc['HasSeizures'] != '_')]
            
            #if there are multiple statements of hasSz or no statements, then print a warning
            if len(set(hasSz_anno_doc['HasSeizures'])) == 1:
                hasSz_spans[doc] = {'text':anno_utils.series_to_paragraph(hasSz_anno_doc['Token']), 'label':"".join([ch for ch in set(hasSz_anno_doc['HasSeizures']).pop() if ch.isalpha()])}
            else:
                print(f"WARNING: Improper HasSz annotation detected at {doc}")
                hasSz_spans[doc] = hasSz_anno_doc
                
    return hasSz_spans

def combine_span_dicts(list_of_spans, label=None):
    #one array to hold the combined list, and the other to get the indices where each group ends and the other begins
    combined_list = []
    group_end_idx = [0]
    used_docs = []
    for spanlist in list_of_spans:
        if label:
            combined_list.append([spanlist[doc_name]['text'] for doc_name in spanlist if spanlist[doc_name]['label'] == label])
            used_docs.append([doc_name for doc_name in spanlist if spanlist[doc_name]['label'] == label])
        else:
            combined_list.append([spanlist[doc_name]['text'] for doc_name in spanlist])
            used_docs.append([doc_name for doc_name in spanlist])
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
        
def calculate_similarity_scores(proj_dir_epi, proj_dir_gen, proj_dir_neuro, use_kernel=True, subset_train=None, subset_test=None):
    #get the annotations    
    if use_kernel:
        epi_train = get_hasSz_annotation_text_and_label(proj_dir_epi, subset=subset_train)
        epi_hasSz = get_hasSz_annotation_text_and_label(proj_dir_epi, subset=subset_test)
        gen_hasSz = get_hasSz_annotation_text_and_label(proj_dir_gen)
        neuro_hasSz = get_hasSz_annotation_text_and_label(proj_dir_neuro)
    else:
        epi_train = exclude_hasSz_annotation_text_and_label(proj_dir_epi, subset=subset_train)
        epi_hasSz = exclude_hasSz_annotation_text_and_label(proj_dir_epi, subset=subset_test)
        gen_hasSz = exclude_hasSz_annotation_text_and_label(proj_dir_gen)
        neuro_hasSz = exclude_hasSz_annotation_text_and_label(proj_dir_neuro)
        
    #create a list that contains all phrases of yes, and one for all phrases of no, and one for idk if applicable
    all_hasSz_yes = combine_span_dicts([epi_train, epi_hasSz, neuro_hasSz, gen_hasSz], 'Yes')
    all_hasSz_no = combine_span_dicts([epi_train, epi_hasSz, neuro_hasSz, gen_hasSz], 'No')
    if not use_kernel:
        all_hasSz_idk = combine_span_dicts([epi_train, epi_hasSz, neuro_hasSz, gen_hasSz], 'Unspecified')        
    
    #calculate all embeddings
    all_hasSz_yes_embeddings = np.array([sentence_mdl.encode(sentence) for sentence in all_hasSz_yes[0]])
    all_hasSz_no_embeddings = np.array([sentence_mdl.encode(sentence) for sentence in all_hasSz_no[0]])
    if not use_kernel:
        all_hasSz_idk_embeddings = np.array([sentence_mdl.encode(sentence) for sentence in all_hasSz_idk[0]])        
        
    #calculate levenshtein similarities pairwise across all documents
    pw_yes_levenshtein = np.array([[Levenshtein.ratio(all_hasSz_yes[0][i], all_hasSz_yes[0][j]) for i in range(len(all_hasSz_yes[0]))] for j in range(len(all_hasSz_yes[0]))])
    pw_no_levenshtein = np.array([[Levenshtein.ratio(all_hasSz_no[0][i], all_hasSz_no[0][j]) for i in range(len(all_hasSz_no[0]))] for j in range(len(all_hasSz_no[0]))])
    if not use_kernel:
        pw_idk_levenshtein = np.array([[Levenshtein.ratio(all_hasSz_idk[0][i], all_hasSz_idk[0][j]) for i in range(len(all_hasSz_idk[0]))] for j in range(len(all_hasSz_idk[0]))])       
            
    #calculate cosine similarities pairwise across all documents
    pw_yes_cos = np.array(util.cos_sim(all_hasSz_yes_embeddings, all_hasSz_yes_embeddings))
    pw_no_cos = np.array(util.cos_sim(all_hasSz_no_embeddings, all_hasSz_no_embeddings))
    if not use_kernel:
        pw_idk_cos = np.array(util.cos_sim(all_hasSz_idk_embeddings, all_hasSz_idk_embeddings))
        
    if use_kernel:
        return all_hasSz_yes, all_hasSz_no, pw_yes_levenshtein, pw_no_levenshtein, pw_yes_cos, pw_no_cos
    else:
        return all_hasSz_yes, all_hasSz_no, all_hasSz_idk, pw_yes_levenshtein, pw_no_levenshtein, pw_idk_levenshtein, pw_yes_cos, pw_no_cos, pw_idk_cos 

def get_grouped_comparisons(all_hasSz, lev_sims, cos_sims, order = [(1,1), (1,2), (1,3), (1,4)], use_kernel=True, remove_outliers=5):
    #hold values in each grouped comparison, split by question
    group_yes_lev = []
    group_no_lev = []
    group_idk_lev = []
    group_yes_cos = []
    group_no_cos = []
    group_idk_cos = []
    
    #hold values in each grouped comparison, regardless of question
    group_all_lev = []
    group_all_cos = []
    
    #keep track of the group comparison labels
    labels=[]
    
    #create groups
    for i,j in order:
        
        #slice into the 2D matrix
        group_yes_lev.append(lev_sims[0][all_hasSz[0][1][i-1]:all_hasSz[0][1][i], all_hasSz[0][1][j-1]:all_hasSz[0][1][j]])
        group_no_lev.append(lev_sims[1][all_hasSz[1][1][i-1]:all_hasSz[1][1][i], all_hasSz[1][1][j-1]:all_hasSz[1][1][j]])
        group_yes_cos.append(cos_sims[0][all_hasSz[0][1][i-1]:all_hasSz[0][1][i], all_hasSz[0][1][j-1]:all_hasSz[0][1][j]])
        group_no_cos.append(cos_sims[1][all_hasSz[1][1][i-1]:all_hasSz[1][1][i], all_hasSz[1][1][j-1]:all_hasSz[1][1][j]])
        if not use_kernel:
            group_idk_lev.append(lev_sims[2][all_hasSz[2][1][i-1]:all_hasSz[2][1][i], all_hasSz[2][1][j-1]:all_hasSz[2][1][j]])
            group_idk_cos.append(cos_sims[2][all_hasSz[2][1][i-1]:all_hasSz[2][1][i], all_hasSz[2][1][j-1]:all_hasSz[2][1][j]])
        labels.append(f"{groups[i-1]} - {groups[j-1]}")

        #if we're on the main diagonal, then we're going to have mirrored items. Remove the duplicates. #we also don't care about the main diagonal itself, so we can remove those too
        if i == j:
            yes_tril_idx = np.tril_indices(group_yes_lev[-1].shape[0], k=0)
            no_tril_idx = np.tril_indices(group_no_lev[-1].shape[0], k=0)
            group_yes_lev[-1][yes_tril_idx] = np.nan
            group_no_lev[-1][no_tril_idx] = np.nan
            group_yes_cos[-1][yes_tril_idx] = np.nan
            group_no_cos[-1][no_tril_idx] = np.nan
            if not use_kernel:
                idk_tril_idx = np.tril_indices(group_idk_lev[-1].shape[0], k=0)
                group_idk_lev[-1][idk_tril_idx] = np.nan
                group_idk_cos[-1][idk_tril_idx] = np.nan

        #now, flatten the arrays and drop nans
        group_yes_lev[-1] = group_yes_lev[-1].flatten()
        group_yes_lev[-1] = group_yes_lev[-1][~np.isnan(group_yes_lev[-1])]
        group_no_lev[-1] = group_no_lev[-1].flatten()
        group_no_lev[-1] = group_no_lev[-1][~np.isnan(group_no_lev[-1])]
        group_yes_cos[-1] = group_yes_cos[-1].flatten()
        group_yes_cos[-1] = group_yes_cos[-1][~np.isnan(group_yes_cos[-1])]
        group_no_cos[-1] = group_no_cos[-1].flatten()
        group_no_cos[-1] = group_no_cos[-1][~np.isnan(group_no_cos[-1])]
        if not use_kernel:
            group_idk_lev[-1] = group_idk_lev[-1].flatten()
            group_idk_lev[-1] = group_idk_lev[-1][~np.isnan(group_idk_lev[-1])]
            group_idk_cos[-1] = group_idk_cos[-1].flatten()
            group_idk_cos[-1] = group_idk_cos[-1][~np.isnan(group_idk_cos[-1])]

        #append values into the containers that hold all similarities
        if use_kernel:
            group_all_lev.append(np.concatenate([group_yes_lev[-1], group_no_lev[-1]]))
            group_all_cos.append(np.concatenate([group_yes_cos[-1], group_no_cos[-1]]))
        else:
            group_all_lev.append(np.concatenate([group_yes_lev[-1], group_no_lev[-1], group_idk_lev[-1]]))
            group_all_cos.append(np.concatenate([group_yes_cos[-1], group_no_cos[-1], group_idk_cos[-1]]))

    #remove outliers in each question group
    group_yes_lev = remove_N_outliers(group_yes_lev, remove_outliers)
    group_yes_cos = remove_N_outliers(group_yes_cos, remove_outliers)
    group_no_lev = remove_N_outliers(group_no_lev, remove_outliers)
    group_no_cos = remove_N_outliers(group_no_cos, remove_outliers)
    if not use_kernel:
        group_idk_lev = remove_N_outliers(group_idk_lev, remove_outliers)
        group_idk_cos = remove_N_outliers(group_idk_cos, remove_outliers)
        
    #remove outliers in combined groups
    group_all_lev = remove_N_outliers(group_all_lev, remove_outliers)
    group_all_cos = remove_N_outliers(group_all_cos, remove_outliers)
    
    if use_kernel:
        return group_yes_lev, group_no_lev, group_yes_cos, group_no_cos, group_all_lev, group_all_cos, labels
    else:
        return group_yes_lev, group_no_lev, group_idk_lev, group_yes_cos, group_no_cos, group_idk_cos, group_all_lev, group_all_cos, labels

def sim_to_train(doc_id, ground_truth):
    """
    Calculates the levenshtein and cosine similarity for a specific doc_id from the all_model_preds table.
    This function uses variables outside the scope of the function.
    doc_id: The doc's ID, as the index of the table
    ground_truth: The ground truth value of this document, for use in identifying which matrix to use
    """
    #prepare to find which index in the similarity matrices to use
    #the doc_id has an extra _hasSz at the end of it. We can remove it by splitting and taking everything but the last one
    doc_name = "_".join(doc_id.split("_")[:-1])
    
    #calculate the averaged document similarities for each document
    #if ground_truth=0, then hasSz = no, and we're using all_hasSz_no, all_hasSz_no_context, pw_no_levenshtein, pw_no_cos, pw_no_levenshtein_context, pw_no_cos_context
    #if ground_truth=1, then hasSz = yes, and we're using all_hasSz_yes, ...
    #if ground_truth=2, then hasSz = IDK, and we only need all_hasSz_idk_context, pw_idk_levenshtein_context, pw_idk_cos_context
    if ground_truth==0:
        #find where doc_name occurs in all_hasSz_no and all_hasSz_no_context
        doc_idx = [idx for idx,x in enumerate(all_hasSz_no[2]) if doc_name in x]
        doc_idx_context = [idx for idx,x in enumerate(all_hasSz_no_context[2]) if doc_name in x]
        
        #find the average similarities this doc has from the epi_train notes
        avg_lev = np.mean(pw_no_levenshtein[all_hasSz_no[1][0]:all_hasSz_no[1][1], doc_idx])
        avg_cos = np.mean(pw_no_cos[all_hasSz_no[1][0]:all_hasSz_no[1][1], doc_idx])
        avg_lev_context = np.mean(pw_no_levenshtein_context[all_hasSz_no_context[1][0]:all_hasSz_no_context[1][1], doc_idx_context])
        avg_cos_context = np.mean(pw_no_cos_context[all_hasSz_no_context[1][0]:all_hasSz_no_context[1][1], doc_idx_context])
        
        return pd.Series({"avg_lev":avg_lev, "avg_cos":avg_cos, "avg_lev_context":avg_lev_context, "avg_cos_context":avg_cos_context})
    elif ground_truth==1:
        #find where doc_name occurs in all_hasSz_yes and all_hasSz_yes_context
        doc_idx = [idx for idx,x in enumerate(all_hasSz_yes[2]) if doc_name in x]
        doc_idx_context = [idx for idx,x in enumerate(all_hasSz_yes_context[2]) if doc_name in x]
        
        #find the average similarities this doc has from the epi_train notes
        avg_lev = np.mean(pw_yes_levenshtein[all_hasSz_yes[1][0]:all_hasSz_yes[1][1], doc_idx])
        avg_cos = np.mean(pw_yes_cos[all_hasSz_yes[1][0]:all_hasSz_yes[1][1], doc_idx])
        avg_lev_context = np.mean(pw_yes_levenshtein_context[all_hasSz_yes_context[1][0]:all_hasSz_yes_context[1][1], doc_idx_context])
        avg_cos_context = np.mean(pw_yes_cos_context[all_hasSz_yes_context[1][0]:all_hasSz_yes_context[1][1], doc_idx_context])
        
        return pd.Series({"avg_lev":avg_lev, "avg_cos":avg_cos, "avg_lev_context":avg_lev_context, "avg_cos_context":avg_cos_context})
    else:
        #find where doc_name occurs all_hasSz_idk_context
        doc_idx_context = [idx for idx,x in enumerate(all_hasSz_idk_context[2]) if doc_name in x]
        
        #find the average similarities this doc has from the epi_train notes
        avg_lev_context = np.mean(pw_idk_levenshtein_context[all_hasSz_idk_context[1][0]:all_hasSz_idk_context[1][1], doc_idx_context])
        avg_cos_context = np.mean(pw_idk_cos_context[all_hasSz_idk_context[1][0]:all_hasSz_idk_context[1][1], doc_idx_context])
        avg_lev = np.nan
        avg_cos = np.nan
        
        return pd.Series({"avg_lev":avg_lev, "avg_cos":avg_cos, "avg_lev_context":avg_lev_context, "avg_cos_context":avg_cos_context})

#where to save the figures
fig_save_dir = 'figures'

#where are the annotations stored?
proj_dir_epi = 'epipleptologist_annotations/' 
proj_dir_neuro = 'neurologist_annotations/'
proj_dir_general = 'non_epileptologist_annotations/'

#where is the generalization test set and the JAMIA training set stored, so we can properly compare apples to apples?
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

#calculate similarity scores for the surrounding contexts
all_hasSz_yes_context, all_hasSz_no_context, all_hasSz_idk_context, pw_yes_levenshtein_context, pw_no_levenshtein_context, pw_idk_levenshtein_context, pw_yes_cos_context, pw_no_cos_context, pw_idk_cos_context = \
    calculate_similarity_scores(proj_dir_epi, proj_dir_gen, proj_dir_neuro, use_kernel=False, subset_train=jamia_train_files, subset_test=epi_test_files)

#calculate similarity scores for the kernels
all_hasSz_yes, all_hasSz_no, pw_yes_levenshtein, pw_no_levenshtein, pw_yes_cos, pw_no_cos = \
    calculate_similarity_scores(proj_dir_epi, proj_dir_gen, proj_dir_neuro, use_kernel=True, subset_train=jamia_train_files, subset_test=epi_test_files)


#create comparison groups
group_yes_lev_context, group_no_lev_context, group_idk_lev_context, group_yes_cos_context, group_no_cos_context, group_idk_cos_context, group_all_lev_context, group_all_cos_context, labels = \
    get_grouped_comparisons([all_hasSz_yes_context, all_hasSz_no_context, all_hasSz_idk_context], \
                            [pw_yes_levenshtein_context, pw_no_levenshtein_context, pw_idk_levenshtein_context], \
                            [pw_yes_cos_context, pw_no_cos_context, pw_idk_cos_context], \
                            use_kernel = False)
group_yes_lev, group_no_lev, group_yes_cos, group_no_cos, group_all_lev, group_all_cos, labels = \
    get_grouped_comparisons([all_hasSz_yes, all_hasSz_no], \
                            [pw_yes_levenshtein, pw_no_levenshtein], \
                            [pw_yes_cos, pw_no_cos], \
                            use_kernel = True)
                            

#iterate through the model predictions. Then, load the model prediction tsv files and get their predictions for each document
#we want to create a table of Doc_ID | Prov Type | Ground truth | Pred seed 2| Pred seed 17 | ... | Pred seed 136
    #we start with a dict of format {DOC_ID: {Prov Type:..., Ground Truth:..., ...}, ...}
all_model_preds = {}
for pred_dir in os.listdir('model_generalization_predictions'):
    
    #skip everything except hasSz predictions
    if 'hasSz' not in pred_dir:
        continue
    
    #load the predictions
    hasSz_pred = pd.read_csv(f"model_generalization_predictions/{pred_dir}/eval_predictions.tsv", sep='\t')
    
    #iterate through the predictions
    for idx, doc in hasSz_pred.iterrows():
        
        #the directory name has some model info in it
        model_info = pred_dir.split("_")
        
        #check if this doc has already been added to the dict
        if doc.ID in all_model_preds: #if it has, then update the entry with this seed's argmax
            all_model_preds[doc.ID][f"seed_{model_info[-1]}"] = doc['argmax']
        else: #if it hasn't then make a new dictionary entry and add this seed's argmax
            all_model_preds[doc.ID] = {
                'prov_type':model_info[1], 
                'ground_truth':doc.True_Label,
                f"seed_{model_info[-1]}":doc['argmax']
            }
all_model_preds = pd.DataFrame(all_model_preds).transpose()

#calculate the overall accuracy of each document. If at least 3 models get the right answer, then by voting they are correct
all_model_preds['overall_acc'] = all_model_preds.apply(lambda x: int(np.round(np.mean([x.seed_2 == x.ground_truth, x.seed_17 == x.ground_truth, x.seed_42 == x.ground_truth, x.seed_97 == x.ground_truth, x.seed_136 == x.ground_truth]))), axis=1)

#for each document, calculate its average levenshtein and cosine similarity to the epileptologist training documents
all_model_preds = all_model_preds.join(pd.DataFrame(all_model_preds.apply(lambda x: sim_to_train(x.name, x.ground_truth), axis=1)))


#Perform Mann Whitney U tests between distributions of similarities - are the distributions significantly different in correct vs incorrect findings?
#isolate the correct and incorrect preds. We ignore idk ground truths
correct_preds = all_model_preds.loc[(all_model_preds.overall_acc == 1) & (all_model_preds.ground_truth != 2)]
incorrect_preds = all_model_preds.loc[(all_model_preds.overall_acc == 0) & (all_model_preds.ground_truth != 2)]

print(f"There are {len(correct_preds)} correct samples, and {len(incorrect_preds)} incorrect samples")
print(f"2S MWU Test for avg_lev: {mannwhitneyu(correct_preds.avg_lev, incorrect_preds.avg_lev)}")
print(f"2S MWU Test for avg_cos: {mannwhitneyu(correct_preds.avg_cos, incorrect_preds.avg_cos)}")
print(f"2S MWU Test for avg_lev_context: {mannwhitneyu(correct_preds.avg_lev_context, incorrect_preds.avg_lev_context)}")
print(f"2S MWU Test for avg_cos_context: {mannwhitneyu(correct_preds.avg_cos_context, incorrect_preds.avg_cos_context)}")


#plot the distributions as box plots. 
# axs.boxplot(group_yes_lev, sym="", positions=positions)
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
axs[0].legend([bp1['boxes'][0], bp2['boxes'][0]], ['Correct Prediction', 'Incorrect Prediction'])
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
plt.savefig(f"{fig_save_dir}/fig_2a.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{fig_save_dir}/fig_2a.pdf", dpi=600, bbox_inches='tight')
plt.show()

#pickle the distribution data for use in combined Q-Q plots with EQA similiarities
all_data_for_export = {'lev_hasSz':group_all_lev,
                       'cos_hasSz':group_all_cos,
                       'lev_context_hasSz':group_all_lev_context,
                       'cos_context_hasSz':group_all_cos_context,
                       'model_preds_hasSz':all_model_preds}
with open('hasSz_similarities.pkl', 'wb') as f:
    pickle.dump(all_data_for_export, f)