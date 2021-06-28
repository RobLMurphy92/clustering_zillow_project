# Zillow_Clustering_Project
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>



#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis utilizing clustering and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Figure out what are driving the errors in the Zestimates?.
> - Utilize clustering to create features for modeling.


#### Goals
> - Find drivers of errors in the Zestimate.
> - Construct a model that accurately predicts log error.
> - Document your process well enough to be presented or read like a report.
> - Create final notebook which will include code and markdowns.

#### Audience
> - Your audience for this project is a data science team.


#### Project Deliverables
> - A clearly named final notebook. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
> - A README that explains what the project is, how to reproduce you work, and your notes from project planning.
> - A Python module or modules that automates the data acquisistion and preparation process. These modules should be imported and used in your final notebook.


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Data Dictionary

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| logerror| float64 | 𝑙𝑜𝑔𝑒𝑟𝑟𝑜𝑟=𝑙𝑜𝑔(𝑍𝑒𝑠𝑡𝑖𝑚𝑎𝑡𝑒)−𝑙𝑜𝑔(𝑆𝑎𝑙𝑒𝑃𝑟𝑖𝑐𝑒)

> - For Data Dictionary please see link below:
> - https://github.com/RobLMurphy92/clustering_zillow_project/blob/master/zillow_data_dictionary.xlsx





<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Planning:

> - Create a README.md which contains a data dictionary, project objectives, business goals, initial hypothesis.
> - Acquire Zillow dataset from the Codeup databse, create a function which will use a sql query and pull specific tables save this function in a acquire.py
> - Prep the Zillow dataset.
> - Investigate missing values and outliers come to a conclusion to either drop or fill these values.
> - Explore the dataset on unscaled data, look into the interaction between independent variables and the target variable using visualization and statistical testing.
> - Clustering utilized to explore the data.
> - Four different models are to be created and compare performance. One model will have the distinct combination of algorithm, hyperparameters, and features.
> - Evaluate the models on the scaled train and validate datasets.
> - Choose the model which performs the best, then run that model on the test dataset.
> - Present conclusions and main takeaways.

#### Hypotheses: 
##### Pre Clustering

> - **Hypothesis 1 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null: 'There is no relationship between logerror and house_age'
> - Hypothesis Alternative : 'There is a relationship between logerror and house_age'


> - **Hypothesis 2 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'There is no relationship between logerror and total_square_ft'
> - Hypothesis Alternative : 'There is a relationship between logerror and total_square_ft'

> - **Hypothesis 3 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'There is no relationship between logerror and latitude'
> - Hypothesis Alternative : 'There is a relationship between logerror and latitude'

> - **Hypothesis 4 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : "There is no significant difference in the logerrors for those who have four or more bedrooms than those who dont."
> - Hypothesis Alternative : "There is a significant difference in the logerrors for those who have four or more bedrooms than those who dont."

> - **Hypothesis 5 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : "There is no significant difference in the logerrors for homes with less than 2500 sqrft than homes with greater than."
> - Hypothesis Alternative : "There is a significant difference in the logerrors for homes with less than 2500 sqrft than homes with greater than."

> - **Hypothesis 6 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : "There is no significant difference in the logerrors for homes greater than 60 years old than homes less than 60 years old"
> - Hypothesis Alternative : "There is a significant difference in the logerrors for homes greater than 60 years old than homes less than 60 years old"


#### Hypotheses: 
##### Post Clustering

> - **Hypothesis 1 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null: 'is no difference in the means of the individual clusters logerrors to the population logerror mean.'
> - Hypothesis Alternative: 'is a significant difference in the means of the clusters logerrors.'


> - **Hypothesis 2 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'is no difference in the means of the individual clusters logerrors to the population logerror mean.
> - Hypothesis Alternative : 'is a difference in the means of the individual clusters logerrors to the population logerror mean.


### Reproduce My Project:

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, explore.py, model.py and final_notebook.ipynb files into your working directory
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_notebook.ipynb 



