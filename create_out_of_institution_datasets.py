import docx
import docx2txt
import json
import argparse
import os
import re

#Input arguments
parser = argparse.ArgumentParser(description="Converts a directory of docx note files into json datasets")
parser.add_argument("--directory", help="The path to the directory with the docx note files", required=True)
parser.add_argument("--save_directory", help="The path to the directory where the datasets will be saved. Datasets will be saved there as hasSz.json, and szFreq.json", required = True)
args = parser.parse_args()

#list the files in the directory
docs =  [doc for doc in os.listdir(args.directory) if '.docx' in doc]

#set up the dataset structures
eqa_dataset = {'version':'v2.0', 'data':[]}
classification_dataset = []
squad_id = 0
max_length = 3*512 - 30
#What are our questions?
classification_q = "Has the patient had recent events?"
eqa_q = ["How often does the patient have events?", "When was the patient's last event?"]
eqa_identifer = ["PQF", "ELO"]

#what text denotes a potential relevant text starting point?
inclusion_regex = r"(?im)( the patient| they| he| she)?( was| were)? last seen"

#open each document and add them to the datasets
for doc in docs:  
    #get the text and sentences of the document
    doc_text = " ".join(docx2txt.process(f"{args.directory}/{doc}").replace('\n', ' ').split())
    sentences = doc_text.split(". ")
    
    #for each sentence, check if they have a matching inclusion phrase, and get the very first one
    regex_matched = False
    for sentence_idx in range(len(sentences)):
        regex_search = re.search(inclusion_regex, sentences[sentence_idx].lower())
        if regex_search:
            regex_matched = True
            break
    
    #if there was an inclusion phrase, then grab text from that sentence and onwards
    #Otherwise, look for specific sections
    found_text = False
    if regex_matched:
        doc_text = ". ".join(sentences[sentence_idx:])[:max_length]
        found_text = True
    else:
        #search for keywords in each sentence, but keep them only if they occur at the very start of the sentence
        for sentence_idx in range(len(sentences)):
            if 'interval history' in sentences[sentence_idx].lower():
                sub_idx = sentences[sentence_idx].lower().index('interval history')
                if sub_idx <= 3:
                    doc_text = ". ".join(sentences[sentence_idx:])[:max_length]
                    found_text = True
                    break
            elif 'interval events' in sentences[sentence_idx].lower():
                sub_idx = sentences[sentence_idx].lower().index('interval events')
                if sub_idx <= 3:
                    doc_text = ". ".join(sentences[sentence_idx:])[:max_length]
                    found_text = True
                    break
            elif 'hpi' in sentences[sentence_idx].lower():
                sub_idx = sentences[sentence_idx].lower().index('hpi')
                if sub_idx <= 3:
                    doc_text = ". ".join(sentences[sentence_idx:])[:max_length]
                    found_text = True
                    break
            elif 'history of present illness' in sentences[sentence_idx].lower():
                sub_idx = sentences[sentence_idx].lower().index('history of present illness')
                if sub_idx <= 3:
                    doc_text = ". ".join(sentences[sentence_idx:])[:max_length]
                    found_text = True
                    break
    #if not regex was matched, and no keywords were found, then 
    if not found_text:
        doc_text = doc_text[0:max_length]
        found_text = True
    
    #add eqa questions
    for i in range(len(eqa_q)):
        eqa_dataset['data'].append({
            'id':doc+"_"+eqa_identifer[i],
            'context':doc_text,
            'question':eqa_q[i]
        })
    
    #add the classification question
    classification_dataset.append({
        'id':doc+"_hasSz",
        'passage':doc_text,
        'question':classification_q
    })
    
#save to files
with open(f"{args.save_directory}/hasSz.json", 'w') as f:
    for datum in classification_dataset:
        json.dump(datum, f)
        f.write('\n')
with open(f"{args.save_directory}/szFreq.json", 'w') as f:
    json.dump(eqa_dataset, f)
