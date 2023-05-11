import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import re
import random
from utils import get_paragraph

#extraction parameters
addendum_string = "I saw and evaluated/examined  on  and reviewed 's notes. I agree with the history, physical exam"
#regex for extracting HPI/Interval History
whitelist_regex = r"(?im)^(\bHPI\b|\bHistory of Present Illness\b|\bInterval History\b)"
blacklist_regex = r"(?im)(\b(Past |Prior )?((Social|Surgical|Family|Medical|Psychiatric|Seizure|Disease|Epilepsy) )?History\b|\bSemiology\b|\bLab|\bExam|\bDiagnostic|\bImpression|\bPlan\b|\bPE\b|\bRisk Factor|\bMedications|\bAllerg)"

#number of notes to extract for the neurologists and non-neurologists (generalists)
num_notes_neuro = 100
num_notes_general = 100

#load the progress and discharge notes
p_notes = pd.read_csv("Progress and Discharge Notes.csv")

#load the provider names, including who is an epileptologist, who is a non-epilepsy specialist neurologist, and non-neurologists (generalists).
prov_info = pd.read_csv('provider_list.csv')
epilepsy_provs = prov_info.loc[prov_info['neurology'] == 'e'][['note_author', 'prov_type']].reset_index(drop=True)
neuro_provs = prov_info.loc[prov_info['neurology'] == 'x'][['note_author', 'prov_type']].reset_index(drop=True)
general_provs = prov_info.loc[pd.isnull(prov_info['neurology'])][['note_author', 'prov_type']].reset_index(drop=True)

#ignore attending attestations and randomize the order
attendings = p_notes['note_text'].apply(lambda x: SequenceMatcher(None, x[:len(addendum_string)], addendum_string).ratio() > 0.60)
p_notes = p_notes[~attendings].sample(frac = 1).reset_index(drop=True)

#use notes that have "seizure" in it
p_notes = p_notes.loc[(p_notes['note_text'].str.contains('seizure')) | (p_notes['note_text'].str.contains('Seizure'))]    
    
#convert dates to datetimes
p_notes['visit_date'] = p_notes['visit_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

#get notes from non-epilepsy neurologists and other providers
p_notes_neuro = p_notes.merge(neuro_provs, on=['note_author', 'prov_type'], how='inner').sample(frac=1).reset_index(drop=True)
p_notes_general = p_notes.merge(general_provs, on=['note_author', 'prov_type'], how='inner').sample(frac=1).reset_index(drop=True)

#parameters for paragraph chunking
splitter = "  "
max_length = 3*512 - 30
hpi_paragraphs_neuro = []
hpi_paragraphs_general = []

#get the neurologist notes
for idx, row in p_notes_neuro.iterrows():
        
    #extract the hpi/interval history relevant paragraphs
    hpi_paragraph_neuro = get_paragraph(whitelist_regex, blacklist_regex, row['note_text'], row['note_author'], row['pat_id'], str(row['visit_date']), splitter, max_length)
    if hpi_paragraph_neuro != None:
        for paragraph in hpi_paragraph_neuro:
            if 'seizure' in hpi_paragraph_neuro[paragraph].lower():
                hpi_paragraphs_neuro.append(hpi_paragraph_neuro)
                break
    
    if len(hpi_paragraphs_neuro) >= num_notes_neuro:
        break
        
#get the non-neurologist notes
for idx, row in p_notes_general.iterrows():
        
    #extract the hpi/interval history relevant paragraphs
    hpi_paragraph_general = get_paragraph(whitelist_regex, blacklist_regex, row['note_text'], row['note_author'], row['pat_id'], str(row['visit_date']), splitter, max_length)
    if hpi_paragraph_general != None:
        for paragraph in hpi_paragraph_general:
            if 'seizure' in hpi_paragraph_general[paragraph].lower():
                hpi_paragraphs_general.append(hpi_paragraph_general)
                break
    
    if len(hpi_paragraphs_general) >= num_notes_general:
        break
        
#save the neurologist notes to files
for paragraph in hpi_paragraphs_neuro:
    filename = paragraph['filename']
    p_idx = [idx for idx in paragraph.keys() if (isinstance(idx, int)& ('seizure' in paragraph[idx].lower()))]
    with open(r"penn_generalization_hpi_extracts/neurologist_notes/"+filename+".txt", 'w') as f:
        f.write(paragraph[random.choice(p_idx)])


#save the non-neurologist notes to files. 
num_saved = 0
for paragraph in hpi_paragraphs_general:
    filename = paragraph['filename']
    p_idx = [idx for idx in paragraph.keys() if (isinstance(idx, int) & ('seizure' in paragraph[idx].lower()))]
    with open(r"penn_generalization_hpi_extracts/general_notes/"+filename+".txt", 'w') as f:
        f.write(paragraph[random.choice(p_idx)])