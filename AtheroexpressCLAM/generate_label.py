#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                                                  Generate Label Datasets")
print("")
print("* Version          : v1.0.0")
print("")
print("* Last update      : 2023-09-18")
print("* Written by       : Yipei (Petra) Song")
print("* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song | Francesco Cisternino.")
print("")
print("* Description      : Generate a balanced and unbalanced dataset of sampleIDs, imageIDs and labels.")
print("")
print("                     [1] https://github.com/MaryamHaghighat/PathProfiler")
print("")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Generate CSV dataset')
# DATA DIRECTORY
parser.add_argument('--csv_input', type=str, default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroexpressCLAM/20231004.CONVOCALS.samplelist.withSMAslides.csv", 
                                                        help='input csv file name including path')
parser.add_argument('--csv_output', type=str, default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroexpressCLAM/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_IPH.csv", 
                                                        help='output csv file name including path')
parser.add_argument('--csv_output_bal', type=str, default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroexpressCLAM/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_IPH_eq.csv", 
                                                        help='balanced output csv file name including path')
parser.add_argument('--h5_dir', type=str, default="/hpc/dhl_ec/VirtualSlides/GLYCC/features_512_dino/h5_files", 
                                                        help='path to h5 files')
parser.add_argument('--classifier', type=str, default="IPH.bin", 
                                                        help='name of variable to use as classifier')
args = parser.parse_args()

# Load source csv
df = pd.read_csv(args.csv_input)

# Create a new dataframe
new_df = pd.DataFrame(columns=["case_id", "slide_id", "label"])

# Traverse the specified directory and match file names
h5_files = os.listdir(args.h5_dir)
for index, row in df.iterrows():
    study_number = str(row["STUDY_NUMBER"])
    matching_files = [f for f in h5_files if study_number in f]
    for file in matching_files:
        slide_id = os.path.splitext(file)[0]
        label = row[args.classifier]
        new_df = new_df.append({
            "slide_id": slide_id,
            "label": label
        }, ignore_index=True)

# Create the case_id column
new_df["case_id"] = new_df["slide_id"].apply(lambda x: x.split('.')[0])

# Remove rows where label is "NA"
new_df = new_df[new_df["label"] != "NA"]

# Save to destination CSV
new_df.to_csv(args.csv_output, index=False)

# create balanced datasets
dataset_unbalanced = new_df

# count cases per category
count_cat_yes = dataset_unbalanced['label'].value_counts(dropna=False)['yes']
count_cat_no = dataset_unbalanced['label'].value_counts(dropna=False)['no']

if count_cat_yes < count_cat_no:
    smaller_count = count_cat_yes
else:
    smaller_count = count_cat_no

# split dataframe into two dataframes
df_yes = dataset_unbalanced[dataset_unbalanced['label'] == 'yes']
df_no = dataset_unbalanced[dataset_unbalanced['label'] == 'no']

# randomly sample n rows from each dataframe
sample_df = pd.concat([df_yes.sample(smaller_count), df_no.sample(smaller_count)])

# randomize the rows of the resulting dataframe
sample_df = sample_df.sample(frac=1)

sample_df.to_csv(args.csv_output_bal, index=False, sep=',')

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+ The MIT License (MIT)                                                                                               +")
print("+ Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song +")
print("+                                                                                                                     +")
print("+ Permission is hereby granted, free of charge, to any person obtaining a copy of this software and                   +")
print("+ associated documentation files (the \"Software\"), to deal in the Software without restriction, including           +")
print("+ without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell             +")
print("+ copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the            +")
print("+ following conditions:                                                                                               +")
print("+                                                                                                                     +")
print("+ The above copyright notice and this permission notice shall be included in all copies or substantial                +")
print("+ portions of the Software.                                                                                           +")
print("+                                                                                                                     +")
print("+ THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT             +")
print("+ LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO           +")
print("+ EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER           +")
print("+ IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR             +")
print("+ THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                          +")
print("+                                                                                                                     +")
print("+ Reference: http://opensource.org.                                                                                   +")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")