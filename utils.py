import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import string
import re

#loads a paragraph of text from a medical note
def get_paragraph(whitelist_regex, blacklist_regex, note_text, note_author, pat_id, visit_date, splitter="  ", max_length=(3*512-30)):
    """Load a paragraph of text given a medical note"""
    
    #split the document into lines
    sentences = note_text.strip('"').split(splitter)
    
    #skip notes that are less than or equal to 5 lines long
    if len(sentences) <= 5:
        return None
        
    #Dictionary to store relevant information of the document
    document = {}
    document['filename'] = f"{pat_id}_{note_author}_{visit_date[:10]}"
    document['note_author'] = note_author
    
    #scan through each line and find indices where it contains a desired header
    whitelist_indices = []
    blacklist_indices = []
    header_indices = []
    for i in range(len(sentences)):
        substr = sentences[i].strip()[:30]
        if re.search(whitelist_regex, substr):
            whitelist_indices.append(i)
            header_indices.append(i)
        elif re.search(blacklist_regex, substr):
            blacklist_indices.append(i)
            header_indices.append(i)
    header_indices.append(-1)     
    
    #if no whitelisted header is found, skip this note
    if len(whitelist_indices) < 1:
        return None
    
    #extract the paragraphs starting from a whitelist header until the next white or blacklisted header. 
    extract_counter = 0
    for i in range(1, len(header_indices)):
        if header_indices[i-1] in blacklist_indices:
            continue
        elif header_indices[i] == -1:
            doc_text = "".join([line.strip() + "\n" for line in sentences[header_indices[i-1]:] if line != ""])[:max_length]
            if len(doc_text) > 250:
                document[extract_counter] = doc_text
                extract_counter += 1
        else:
            doc_text = "".join([line.strip() + "\n" for line in sentences[header_indices[i-1]:header_indices[i]] if line != ""])[:max_length]
            if len(doc_text) > 250:
                document[extract_counter] = doc_text
                extract_counter += 1
    
    #return if there are any extracted paragraphs    
    if len(document) > 2:
        return document
        
def create_huggingface_dataset_from_anno(anno_docs):
    """
    Create a dataset formatted for huggingface from annotations
    """
    #create a 1-hot-encoding dictionary, assuming True, False, no-answer. Yes and No
    #map equally to true and false. 
    #if -1, there was an error in the annotation process. map as no-answer
    one_hot = {"true":1, "false":0, "no-answer":2, 'yes':1, 'no':0, '-1':2}
    
    eqa_dataset = {'version':'v2.0', 'data':[]}
    classification_dataset = []
    
    #for each document
    for doc in anno_docs:
        
        #get the text
        doc_text = doc.get_raw_text()

        #add hasSz annotation as a boolq blank
        classification_datum = {
            'label':None,
            'id':f"{doc.name[:-4]}_hasSz",
            'question':hasSz_Q,
            'passage':doc_text
        }
        
        #add pqf annotation as a blank
        pqf_datum = {
            'answers':{'answer_start':[], 'text':[]},
            'id':f"{doc.name[:-4]}_pqf",
            'question':pqf_Q,
            'context':doc_text,
        }
        
        #add elo annotation as a blank
        elo_datum = {
            'answers':{'answer_start':[], 'text':[]},
            'id':f"{doc.name[:-4]}_elo",
            'question':elo_Q,
            'context':doc_text,
        }

        #for each annotation, add annotations as the answer to each question
        for annotation in doc.annotations:
            #if the annotation is for HasSz,
            if annotation.layer == "HasSeizures":
                classification_datum['label'] = one_hot[str(annotation.get_raw_value() if annotation.get_raw_value() != 'Unspecified' else 'no-answer').lower()]
            #else, for szfreq or date of last occurrence
            elif annotation.layer == 'SeizureFrequency':
                #for date of last occurrences
                if annotation.get_raw_value() == 'ExplicitLastOccurrence':
                    elo_datum['answers']['text'].append(annotation.get_text())
                    elo_datum['answers']['answer_start'].append(doc_text.find(annotation.get_text()))
                #for frequencies
                else:
                    pqf_datum['answers']['text'].append(annotation.get_text())
                    pqf_datum['answers']['answer_start'].append(doc_text.find(annotation.get_text()))

        #add the paf and elo datum to the dataset
        eqa_dataset['data'].append(pqf_datum)
        eqa_dataset['data'].append(elo_datum)

        #add this document to the boolqa dataset
        classification_dataset.append(classification_datum)
        
    return classification_dataset, eqa_dataset

def collect_annotations(annotation_dir, inclusion_names = None, return_annotators=False):
    """
    Collects annotations from the specified annotation_dir, including only those in inclusion_names if passed
    """
    #list all annotation directories. Inside each directory will be the tsv files for each annotator that has performed an annotation
    #if inclusion_names was passed, use only those that were included
    if inclusion_names:
        anno_docs = [doc_name for doc_name in os.listdir(annotation_dir) if doc_name in inclusion_names]
    else:
        anno_docs = os.listdir(annotation_dir)
    num_files = len(anno_docs)
    
    #tabulate all annotators, get the number of documents each have done
    annotator_info = {}
    annotator_to_index = {}
    num_annotators = 0

    #iterate through documents and get annotations
    for doc in anno_docs:
        doc_dir = annotation_dir+"/"+doc
        anno_files = os.listdir(doc_dir)

        #iterate through annotations and get basic info of annotators
        for anno in anno_files:

            username = anno[:-4]

            #add a new annotator if necessary
            if username not in annotator_info:
                annotator_info[username] = 0
                annotator_to_index[username] = num_annotators
                num_annotators += 1

            annotator_info[username] += 1

    index_to_annotator = {v:k for k,v in annotator_to_index.items()}

    print(annotator_info)
    print(annotator_to_index)
    print(index_to_annotator)
    
    #create annotators
    all_annotators = []
    for i in range(num_annotators):
        all_annotators.append(anno_utils.Annotator(index_to_annotator[i], i))
        
    #containers for annotations    
    base_cols = ['Sen-Tok', 'Beg-End', 'Token']
    all_documents = []

    #collect annotations for each Document
    for doc in anno_docs:
        #find the document to be annotated and the annotators
        doc_dir = annotation_dir+"/"+doc
        anno_files = os.listdir(doc_dir)

        #create a new Document
        new_doc = anno_utils.Document(doc)

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
            #rename columns based off of annotator index
            anno_doc = anno_doc.rename(columns={'HasSeizures':'HasSeizures_'+str(annotator_to_index[anno[:-4]]), \
                                               'SeizureFrequency':'SeizureFrequency_'+str(annotator_to_index[anno[:-4]]), \
                                               'TypeofSeizure':'TypeofSeizure_'+str(annotator_to_index[anno[:-4]])})        
            annotations[annotator_to_index[anno[:-4]]] = anno_doc
            all_annotators[annotator_to_index[anno[:-4]]].add_document(new_doc)


        #skip if no annotations have been performed
        if not bool(annotations):
            print("Missing Annotations: " + str(doc))
            continue;


        #combine annotations into a single table
        annotations = reduce(lambda df1, df2: pd.merge(df1, df2, how='outer', 
                                                       on=['Sen-Tok', 'Beg-End', 'Token']), annotations.values())
        #replace empty values
        annotations = annotations.fillna('_')
        #replace incomplete annotations with blanks
        annotations = annotations.replace(to_replace=r'\*.+|\*', value='_', regex=True)    

        #add text to the new Document
        new_doc.set_text(annotations[['Beg-End','Token']])
        #add Document to document container
        all_documents.append(new_doc)

        #process annotations
        anno_utils.collect_annotations(annotations.drop(base_cols,axis=1), new_doc, all_annotators)
        
    if return_annotators:
        return all_documents, all_annotators
    else:
        return all_documents

def save_datasets(classification_dataset, eqa_dataset, classification_path, eqa_path):
    with open(classification_path, 'w') as f:
        for datum in classification_dataset:
            json.dump(datum, f)
            f.write('\n')
    with open(eqa_path, 'w') as f:
        json.dump(eqa_dataset, f)
        
def combine_individual_and_curated_annotations(proj_dir):
    """
    Combines individual annotations and curated annotations into the same directories
    """
    #create a new subdirectory called combined 
    if "combined" not in os.listdir(proj_dir):
        os.mkdir(proj_dir+"/combined")
        
    #populate the combined dirctory with subdirectories of annotation and curation
    for curated_dir in os.listdir(proj_dir+"/curation"):
        os.mkdir(proj_dir+"/combined/"+curated_dir)

        #copy the curated annotations over
        copyfile(proj_dir+"/curation/"+curated_dir+"/CURATION_USER.tsv", proj_dir+"/combined/"+curated_dir+"/CURATION_USER.tsv")

        #for each annotation file in the annotation subdirectory
        for anno_file in os.listdir(proj_dir+"/annotation/"+curated_dir):
            if 'ipynb_checkpoints' in anno_file:
                continue
            copyfile(proj_dir+"/annotation/"+curated_dir+"/"+anno_file, proj_dir+"/combined/"+curated_dir+"/"+anno_file)
            
    print("Done")
    
def compare_annotators(all_annotators):
    #for each annotator, compare them to the others and identify overlapping annotations
    for i in range(len(all_annotators)):
        for j in range(i+1, len(all_annotators)):

            #iterate through the annotations of each annotator
            for anno_1 in range(len(all_annotators[i].annotations)):
                for anno_2 in range(len(all_annotators[j].annotations)):
                    if not all_annotators[i].annotations[anno_1].check_overlap(all_annotators[j].annotations[anno_2]):
                        continue
                    all_annotators[i].annotations[anno_1].add_overlap(all_annotators[j].annotations[anno_2])
                    all_annotators[j].annotations[anno_2].add_overlap(all_annotators[i].annotations[anno_1])
                    all_annotators[i].annotations[anno_1].info()
                    all_annotators[j].annotations[anno_2].info()
                    
def calculate_annotator_agreement_metrics(all_annotators, print_output=True):
    """
    Calculate the pairwise agreement between annotators of all_annotators
    If print_output=True, then print out the metrics
    """
    all_iaa = {}
    #iterate through annotators pairwise
    for i in range(len(all_annotators)):
        for j in range(i+1, len(all_annotators)):   
            if not all_annotators[i].annotations or not all_annotators[j].annotations:
                continue
            
            #calculate agreement
            iaa = anno_utils.Agreement(all_annotators[i], all_annotators[j])
            iaa.calc_simple_agreement()
            cohen_hasSz = iaa.calc_cohen_kappa('HasSeizures')
            cohen_szFreq = iaa.calc_cohen_kappa('SeizureFrequency')
            f1_szFreq = iaa.calc_average_F1_overlap('SeizureFrequency')
            
            #store agreement into all_iaa
            all_iaa[f"{all_annotators[i].name} | {all_annotators[j].name}"] = {
                'iaa':iaa, 
                'cohen_hasSz':cohen_hasSz,
                'cohen_szFreq':cohen_szFreq,
                'f1_szFreq':f1_szFreq
            }
            
            
            if print_output:
                print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                print("Calculating agreement between " + all_annotators[i].name + " and " + all_annotators[j].name)
                print()
                print("HasSz Agreement: ")
                print("Cohen's Kappa: " + str(cohen_hasSz))
                print()
                print("SzFreq Agreement: ")
                print("Cohen's Kappa: " + str(cohen_szFreq)) 
                print("F1 Overlap: " + str(f1_szFreq))
                print()
                iaa.info()
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            
    return all_iaa
    
def load_boolq_formatted_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def load_squad_formatted_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)