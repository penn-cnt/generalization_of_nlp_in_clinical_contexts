import pandas as pd
import json
import numpy as np
import csv
import os
import annotation_utils as anno_utils
from utils import create_huggingface_dataset_from_anno, collect_annotations, save_datasets
from functools import reduce
from sklearn.model_selection import train_test_split

#what is the exported annotation project's local directory?
proj_dir_epi = 'epipleptologist_annotations/' 
proj_dir_neuro = 'neurologist_annotations/'
proj_dir_general = 'non_epileptologist_annotations/'

#where are we saving the QA datasets
epi_train_classification = 'penn_generalization_eval_datasets/train_classification_epi.json'
epi_train_eqa = 'penn_generalization_eval_datasets/train_eqa_epi.json'
eqa_epi_save = 'penn_generalization_eval_datasets/eqa_epi.json'
eqa_gen_save = 'penn_generalization_eval_datasets/eqa_general.json'
eqa_neuro_save = 'penn_generalization_eval_datasets/eqa_neuro.json'
classification_epi_save = 'penn_generalization_eval_datasets/classification_epi.json'
classification_gen_save = 'penn_generalization_eval_datasets/classification_general.json'
classification_neuro_save = 'penn_generalization_eval_datasets/classification_neuro.json'

#What are our questions?
hasSz_Q = "Has the patient had recent events?"
pqf_Q = "How often does the patient have events?"
elo_Q = "When was the patient's last event?"

#The epileptologist datasets wre from our previous JAMIA paper. We need to identify which datum were in the testing and training sets of that paper.
epi_test_set_path = 'JAMIA/hasSz_test.json'
with open(epi_test_set_path, 'r') as f:
    epi_test_set_doc_names = [json.loads(line.rstrip())['title'] for line in f]

#find which documents in the epileptologist set from the JAMIA paper were used in the train set
epi_train_set_path = 'JAMIA/hasSz_train_1.0.json'
with open(epi_train_set_path, 'r') as f:
    epi_train_set_doc_names = [json.loads(line.rstrip())['title'] for line in f]
    
#get the annotations
epi_anno_docs_test = collect_annotations(proj_dir_epi, epi_test_set_doc_names)
epi_anno_docs_train = collect_annotations(proj_dir_epi, epi_train_set_doc_names)
neuro_anno_docs = collect_annotations(proj_dir_neuro)
gen_anno_docs = collect_annotations(proj_dir_general)

#convert annotations to datasets
epi_classification_test, epi_eqa_test = create_huggingface_dataset_from_anno(epi_anno_docs_test)
epi_classification_train, epi_eqa_train = create_huggingface_dataset_from_anno(epi_anno_docs_train)
neuro_classification, neuro_eqa = create_huggingface_dataset_from_anno(neuro_anno_docs)
gen_classification, gen_eqa = create_huggingface_dataset_from_anno(gen_anno_docs)

#save the datasets
save_datasets(epi_classification_test, epi_eqa_test, classification_epi_save, eqa_epi_save)
save_datasets(epi_classification_train, epi_eqa_train, epi_train_classification, epi_train_eqa)
save_datasets(neuro_classification, neuro_eqa, classification_neuro_save, eqa_neuro_save)
save_datasets(gen_classification, gen_eqa, classification_gen_save, eqa_gen_save)