# FeatCompare: Feature Comparison for Competing Mobile Apps Leveraging User Reviews

REQUIRED PROJECTS
-----------------
The below two projects are required to run FeatCompare. Git clone the entire repo into your local directory and install the required packages

1. Please visit (https://github.com/madrugado/Attention-Based-Aspect-Extraction) to clone and run ABAE
2. Please visit (https://github.com/jinyyy666/AR_Miner) to clone and run AR-Miner


FILES DESCRIPTION
-----------------

1. data folder contains the sample data of user reviews
2. scripts folder contains the python script "local_global_selector" used to select the local global reviews. 

DATASET
-------

1. Each file in "app_groups" folder contains the labelled reviews of the groups used in our experiments. The labels are the final ones agreed upon among the annotators.

GENERAL STEPS TO RUN FEATCOMPARE
--------------------------------

For the Local and Global cycles of ABAE, the input data used for training is the same (i.e, user reviews of every group).
The embedding used is different in the Local and Global cycles

1. Run AR-Miner to filter out non informative reviews
2. Run the Global cycle of ABAE
3. Run the Local cycle of ABAE
4. Run the "local_global_selector" script to assign local or global labels to reviews

INSTRUCTION TO RUN ABAE LOCAL
-----------------------------

1. Preprocess and generate the word embedding "local embedding" of the reviews of the specific app groups
2. Use the "local embedding" as input (i.e, --emb-name) to run the train command
3. Use the user reviews of the specific app group as the data input 

INSTRUCTION TO RUN ABAE GLOBAL
------------------------------

1. Preprocess and generate the word embedding "global embedding" of all the reviews of all the apps groups
2. Use the "global embedding" as input (i.e, --emb-name) to run the train command
3. Use the user reviews of the specific app group as the data input
