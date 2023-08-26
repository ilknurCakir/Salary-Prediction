# Salary Prediction

- Project **Salary Prediction** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Project is to predict whether income exceeds $50K/year based on census data.
The project consists of 2 part:
- Developing a model
- Developing an api and deploying it.
The focus is on the second part for the project. 

## Files and data description

The data census.csv is from UCi Data Repository. It can be found on 
https://archive.ics.uci.edu/dataset/20/census+income. It consists of 48842 people and 14 features like 
age, workclass, fnlgt, education, education years, marital status. It is imbalanced dataset that is
only 25% of data is greater than 50K/year.

Files are as follows:

- main.py: It is the script to preprocess data, train the pipeline and generate artifacts
- app.py: It is for the creation of app. 
- tests/: It is a directory that includes all files for unit tests.
- model/: It s a directory that includes all files related to data preprocessing and training of the model.
- model_api/: It is a directory that includes all files related to api creation.
- logger.py: It does initialization of logger
- transformers.py: It includes custom transformer to handle categorical variables
- utils.py: It mainly includes functions to generate sample data for testing purposes
- requirements.txt: List of requirements
- config.yml: It is a file that gives some parameters in the model
- config/core.py: It reads config.yml and stores all parameters in variable config
- README.md: It gives general information about project, data and how to run scripts
- model_card.md: It gives an idea about data, some model specifications, evaluations and ethical considerations.
- live_post.py: It does post to live service.
- logs.log: It includes all loggings. Structure is as date - time - log level - name of the file - message

## Running Files

Environment setup:
Pyhon 3.9 is used in this project
Create a virtual environment and run the following command in the terminal
pip install -r requirements.txt

To train a model and create artifacts, run
python main.py

It runs all steps in sequence. It creates relevant directories and relevant artifacts
under these directories.
All logs will be seen on the terminal and log file.

To run unit tests, run
pytest


It will run all unit tests. If test is a success, it is represented as dot. If it is a 
fail, then it is represented as 'F'. There are a total of 12 unit tests in the file. All tests
should pass and expected result on the terminal is


tests\test_api.py .....                                                  [ 41%]
tests\test_functions.py .......                                          [100%]

============================= 12 passed in 3.42s ==============================


Relevant log messages can be found in log file.

## References

I tried to follow best practices from this repo: https://github.com/trainindata/deploying-machine-learning-models
It is a repo of a Udemy course related to deploying ML models by Chris Samiullah
