# Predicting Food Insecurity

I created this README as a TLDR for my notebook with my main findings. Please ensure you run and explore the interactive graphs on the notebook and view my iterative process in building the final model.

### Credit 

Feeding America provides the target datasets. https://www.feedingamerica.org/

### Installing Dependencies

- run `pipenv install`.

### Jupyter Notebook on GitHub Pages

View the Jupyter Notebook on your web browser at this address: https://classaccounts.github.io/Predicting-Food-Insecurity/

*Note: Plotly Express graphics will not display in HTML form*

## Topic

I am trying to address food insecurity in the United States and identify geographic areas where food programs like the Supplemental Nutrition Assistance Program (SNAP), Special Supplemental Nutrition Program for Women (WIC),  National School Lunch Program (NSLP), and Emergency Food Assistance Program (TEFAP) could be targeted. Those organizations allocate funding based on geographic areas with the greatest need. Therefore, it is essential to address this issue to ensure the right communities get the proper support through these programs to alleviate many health disparities caused by poor nutrition and food insecurity (FI).

*Oxford defines food insecurity as: "The state of being without reliable access to a sufficient quantity of affordable, nutritious food."*

*The USDA defines food insecurity as: "A lack of consistent access to enough food for every person in a household to live an active, healthy life."*

## Project Questions

* Which counties of the United States are most impacted by food insecurity, and what indicators like unemployment, poverty, and income are correlated to it?
* How can we predict food insecurity in the future?

## What would an answer look like?

The essential deliverable would be a choropleth map showing each country's food insecurity index (FII) rate in the United States. I will also have supporting line charts showing the FII rates compared to other indicators like poverty and unemployment. The great thing about relating my dataset using Federal Information Processing Standards (FIPS) is that I can incorporate more datasets to correlate them to the FII of each county. Therefore, I believe I could have multiple answers to my question depending on what indicator correlated to FII we want to look at through exploratory data analysis. This would allow me to create very concrete answers to my questions. Eventually, I will predict food insecurity based on these indicators. 

## Data Sources

I have identified and imported four datasets for this project. However, I decided only to merge three of them into my combined dataset.

* Map the Meal Gap (MMG)
* County Business Patterns (CBP)
* Local Area Unemployment Statistics (LAUS)
* Small Area Income and Poverty Estimates (SAIPE)

These datasets can be related using the FIPS code, which can indicate the county each row pertains to. All of these datasets I have imported contain a FIPS code. Therefore, I can join each of them using the county segment of the FIPS code.

I will explore their relations in this notebook's Exploratory Data Analysis (EDA) section. However, I will need to clean and merge the datasets first.

Please see more information regarding the datasets in the Jupyter notebook.

## EDA

When I merged the datasets, I plotted a correlation matrix of all the features from my combined dataset to which features I would want to start with building my model.

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/corr_matrix.png?raw=true)

I also plotted the mean value of the most correlated features to see if they have a linear relationship and used them as part of my regression model. The features are scaled to the same magnitude to identify their trends better. Please note that the unemployment rate spikes, but food insecurity decreases. This spike was due to the COVID-19 pandemic. You would expect food insecurity also to increase, but it did not follow the previously established linear relationship because of emergency supplemental programs.

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/scaled_mean_value_trends.png?raw=true)

## Modeling & Results 

I successfully created a Machine Learning model that can predict food insecurity rates for a given county in the United States and answer my project questions. I will demonstrate this with the following choropleth maps. To see how I reached these results, please run my notebook and walk through each step from data preparation/cleaning, EDA, to ML modeling.

The interactive graphs in the results section of my notebook answer the following project questions:

* *Which counties of the United States are most impacted by food insecurity, and what indicators like poverty, unemployment, and income are correlated to it?*
* *How can we predict food insecurity in the future?*

Please check out the interactive maps on the notebook so you can view specific county food insecurity rates.

### 2020 Actual Food Insecurity Map

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_actual.png?raw=true)
### 2020 Machine Learning Model Predicted Food Insecurity Map

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_prediction.png?raw=true)
### 2020 Machine Learning Model Predicting Food Insecurity Map Without 2020 Training Data
Since the previous map predicted 2020 actuals based on trained data from 2010 - 2020, I also wanted to create another trained model (2010 - 2019) that was not fitted with 2020 data and see how it predicts 2020 food insecurity, and you can see the results below. However, it does tend to overestimate counties with low food insecurity and predicts them to be higher than their actual (dark blue). I believe this is due to the unemployment spike from the COVID pandemic (2020), where food insecurity followed the mean yearly trend while unemployment rose and in which training data relating to that was purposely removed from this model. The regression model above would do a better job of predicting food insecurity with the COVID unemployment spike. It is challenging to create a model that predicts food insecurity given the COVID pandemic, emergency governmental support programs, and how it adjusted poverty and unemployment trends related to food insecurity. I would eventually like to adjust for this bias in further development.

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_prediction_no2020train.png?raw=true)
