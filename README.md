# London-Fire-Brigade

#### Project Status: [Completed as of March 6th, 2022]

## Project Intro/Objective
The purpose of this project is to analyze emergencies in London, England and give the probability of an emergency being real and not a false alarm. The potential impact is that the london fire brigade can save resources by not responding to emergencies that are very likely a false alarm. 

### Methods Used
* Data Visualization
* Machine Learning
  * Logistic Regression
  * Bayesian Regression
  * Naive Bayes
  * Linear SVC
  * Random Forest
  * Extremely Random Tress
  * AdaBoost
  * XGBoost

### Technologies 
* Python (pandas, numpy, matplotlib)
* Google BigQuery


## Project Description
The raw data comes from Google BigQuery. The whole purpose is to reduce the number of false alarm calls the london fire brigade responds to. Therefore, various incident details and their relationship to the incident status were examined. The incident status originally contained three values: false alarm, special service, and fire. Since the fire brigade needs to repond to both fires and special services, these two values were combined into one value labeled emergency. Since it is much more serious if the fire brigade does not respond to a true emergency, false negative, the recall score was used as the performance metric. The features used included: month, hour, property category, address qualifier, borough name, pump arrival time, number of station with pumps, and number of pumps. Numerous different models were used to get the best score. Overall, Logistic Regression was found to have the best recall score of 86%. Finally, the best model was saved and deployed into production by looking at its predicted probabilities based on randomly generated incident details.


## Getting Started

1. Fork and then Clone this repo
2. Raw Data is kept at Google BigQuery. 
    * To access, you'll need to setup a free account. You can do so [here](https://cloud.google.com/bigquery?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-p-dr-1011347&utm_content=text-ad-none-any-DEV_c-CRE_573203586659-ADGP_Desk%20%7C%20BKWS%20-%20PHR%20%7C%20Txt%20~%20Data%20Analytics%20~%20BigQuery_Big%20Query-KWID_43700068782255081-aud-388092988201%3Akwd-300487425311&utm_term=KW_google%20bigquery-ST_google%20bigquery&gclsrc=aw.ds&gclid=CjwKCAiA6seQBhAfEiwAvPqu14q7OKCAazrTELuSwrUZqDBUd5QNdgdYN1inSrmcY65-yGdVj_XAPRoCr6kQAvD_BwE). 
    * After you'll need to change to project id string used in the fetch london data function.
4. Data processing/transformation scripts/modules are kept in this repository.

## Featured Notebooks
* [Data Cleaning](https://github.com/AustinYunker/London-Fire-Brigade/blob/main/Data%20Cleaning.ipynb)
* [Data Visualization](https://github.com/AustinYunker/London-Fire-Brigade/blob/main/Data%20Visualizations.ipynb)
* [Models](https://github.com/AustinYunker/London-Fire-Brigade/blob/main/London%20Models.ipynb)
* [Productionizing the Model](https://github.com/AustinYunker/London-Fire-Brigade/blob/main/Productionize%20Model.ipynb)


## Contact
* Feel free to contact me with any questions or if you are interested in contributing!

## Acknowledgements 
* This style of README was adopted from Rocio Ng and the original template can be found [here](https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md)
