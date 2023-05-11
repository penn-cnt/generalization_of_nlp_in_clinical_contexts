import pandas as pd
import json
import numpy as np
import csv
import os
import annotation_utils as anno_utils
from utils import collect_annotations, combine_individual_and_curated_annotations, compare_annotators, calculate_annotator_agreement_metrics
from shutil import copyfile
from functools import reduce
import matplotlib.pyplot as plt

#what is the exported project's local directory?
proj_dir_gen = 'non_epileptologist_annotations/'
proj_dir_neuro = 'neurologist_annotations/'

#created the combined annotation directories
combine_individual_and_curated_annotations(proj_dir_gen)
combine_individual_and_curated_annotations(proj_dir_neuro)

#process the annotations
gen_annos, gen_annotators = collect_annotations(f"{proj_dir_gen}combined", return_annotators=True)
neuro_annos, neuro_annotators = collect_annotations(f"{proj_dir_neuro}combined", return_annotator=True)

#identify overlapping annotations between annotators
compare_annotators(gen_annotators)
compare_annotators(neuro_annotators)

#calculate annotator metrics
gen_iaa = calculate_annotator_agreement_metrics(gen_annotators, False)
neuro_iaa = calculate_annotator_agreement_metrics(neuro_annotators, False)

#print results
print(neuro_iaa)
print(gen_iaa)