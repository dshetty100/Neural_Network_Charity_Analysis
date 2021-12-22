# Neural Network Charity Analysis
Charity organization analysis using neural networks

## Overview of the Analysis
The purpose of this analysis was to create a binary classifier that can predict whether applicants to Alphabet Soup, a charity organization, will be successful with the funding that is provided to them. The analysis was carried out using neural network, and a dataset (charity_data.csv) that contained more than 34,000 organizations that have received funding from Alphabet Soup over the years.

 The dataset included the following columns:

* **EIN** and **NAME** — Identification columns
* **APPLICATION_TYPE** — Alphabet Soup application type
* **AFFILIATION** — Affiliated sector of industry
* **CLASSIFICATION** — Government organization classification
* **USE_CASE** — Use case for funding
* **ORGANIZATION** — Organization type
* **STATUS** — Active status
* **INCOME_AMT** — Income classification
* **SPECIAL_CONSIDERATIONS** — Special consideration for application
* **ASK_AMT** — Funding amount requested
* **IS_SUCCESSFUL** — Was the money used effectively

The libraries used were:
* Pandas
* Scikit-learn
* Tensorflow

The goal of the analysis was to utilize the Scikit-learn and Tensorflow libraries to develop a model 
that can be over 75% accurate in predicting whether the applicant will be successful with the funding provided to them by Alphabet Soup.


## Results
### Preprocessing
The first step in the analysis was to examine and preprocess the dataset. 
* The target variable was the **IS_SUCCESSFUL** column.
* The columns **EIN** and **NAME** in the dataset were unnecessary for the purposes of this model are were dropped from the dataset. 
* All other columns were considered potential features for the model.

The next steps in the analysis were to bin, encode, and scale the data.
* Any **APPLICATION_TYPE** with less than 1000 entries were binned into **OTHER**.
* Any **CLASSIFICATION** with less than 1000 entries were binned into **OTHER**.
* All "object" type columns were encoded using OneHotEncoder.
* The **SPECIAL_CONSIDERATIONS_N** column was dropped, as it was redundant to the **SPECIAL_CONSIDERATIONS_Y** column.
* The preprocessed data was then split into training and testing datasets.
* All columns were then scaled using SciKit-Learn's StandardScaler.
* 

### Compiling, Training, and Evaluating

Initially, the model was configured to have the following:
* two hidden layers -- one with 80 neurons, the second with 45 neurons. This provided with 6,891 total and trainable parameters;
* both hidden layers used 'relu' activation functions;
* the output layer used 'sigmoid' activation function.
This resulted in only 72.6% accuracy for the prediction for this model.

In an attempt to optimize and reach 75% accuracy three different models were built by making the following changes to the first model.
* binning **INCOME_AMT** values greater than $5 million into a '5M+' bin;
* adding a third hidden layer;
* increasing the total number of trainable parameters to as high as 9,411;
* increasing training epochs from 100 to 150, then as high as 300;
* trying out both the 'adamax' and 'nadam' optimizers when compiling the model;
* using 'tanh' activation functions on the hidden layers;
* un-binning certain values by lowering the threshold from 1000 values to 700 values on both **APPLICATION_TYPE** and **CLASSIFICATION**.
* the results were saved to an HDF5 file

Unfortunately, all four models were unable to raise the models' accuracy above 73%.

## Summary
Neural networks are complex machines. There are lots of moving part to keep track of. There are also lots of "black boxes" which make analyzing both the processes and results into a hazy, vague endeavor. I'm honestly not sure what would help improve accuracy on these models, except for a lot of brute-force experimentation. Increasing parameters or training epochs don't seem to be that effective. Perhaps attempting some other methods, such as random forests or SVM might yield better results. Maybe there are outliers which I just was unable to pin-point in my examinations. That said, 73% accuracy isn't that bad in the world of finance. Could be better -- could be a lot worse.