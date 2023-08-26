# Model Card - Predicting if the person makes less or more than 50K

## Model Details
* The model was created by I.Cakir
* It is SGD Classifier with modified huber loss using mostly default parameters from scikit-learn
* Binary classification if people make less than or more than 50K

## Intended Use
* The model should be used for applications where you need the information if person makes less or larger than 50K based on personal attributes
* Use cases might include loan applications and dating applications where you can predict candidate date matches' salary

## Training Data
* Data is Census Income data extracted in 1994. It can be obtained from UCI ML Repository https://archive.ics.uci.edu/dataset/20/census+income
* Data was split into training and test data at 3:1 ratio

## Evaluation Data
* Evaluation is done on test data set that is 1/4 of the entire dataset.

## Metrics
* Evaluation metrics include precision, recall and f1 score.
* They are 0.68347, 0.42916 and 0.52726, respectively.

## Ethical Considerations
* It is good that data is anonymous, ie. there is no PII in it.
* Bias is present in supervised and unsupervised level. This implies unfairness in training data and model as well.

## Caveats and Recommendations
* An ideal dataset would additionally include annotations for from people with different job groups and different countries as native country. We don't see many different job groups.
* Sex includes only male and female. Further work is required to evaluate based on broad gender spectrum.
