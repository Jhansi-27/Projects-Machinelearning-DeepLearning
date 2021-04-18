<h2>Travel Insurance Assignment</h2>

```Description of prediction task and datset provided```
A travel insurance company would like to offer a discounted premium (for the same cover requested) to customers that are less likely to make a future claim. The manager contacts you to investigate the feasibility of using machine learning to predict whether a customer will file a claim on their travel. The manager has access to historical data of past policies and she offers to provide you with information about the insured, the purpose and destination of the travel, and whether a
claim was filed.

```The following are the objectives that need to be fulfilled for this assignment```
<ol>
<li>Investigate the performance of a number of machine learning procedures on this
dataset. Using the data in the file CE802_P2_Data.csv, perform a comparative study of the following machine learning procedures:
- a Decision Tree classifier;
- at least two more ML technique to predict if the insured will file a claim.</li>
<li> One of the features, is missing for some of the instances. You are therefore
required to deal with the problem of missing features before you can proceed with the prediction step.
As a baseline approach you may try to discard the feature altogether and train on the remaining
  features. You are then encouraged to experiment with different inputation methods.</li>
<li>The company uses Python internally and therefore Python with scikit-learn is the required
language and machine learning library for the problem. For this task, you are expected to submit a
Jupyter Notebook containing the Python code used to perform
the comparative analysis and produce the results, as well as the code used to perform the predictions
  described in task “b” below.</li>
<li> ```task b:``` Prediction on a hold-out test set. An additional dataset, CE802_P2_Test.csv, is provided. Binary outcomes are withheld for this test set (i.e. the
“Class” column is empty). In this second task you are required to produce class predictions of
the records in the test set using one approach of your choice among those tested in task “a” (for
example the one achieving the best performance). These data must not be used other than to test
  the algorithm trained on the training data.</li>
<li>As part of your submission you should submit a new version of the file CE802_P2_Test.csv in
CSV format with the missing class replaced with the output predictions obtained using the approach
  chosen. This second task will be marked based on the prediction accuracy on the test set.</li></ol>
3. Additional Comparative Study
Thanks to the good results obtained in the comparative study, the company has deployed your
system and is obtaining good profit. Now a competitor would like to hire you to design a similar
system for them but, unlike the first system, they would like you to predict not only if the insured
files a claim but also the value of the claim.
They provide you with a training set of historical data containing features of each customer
and a numerical value representing the value of the claim (which may be zero). These data are
available in the CE802_P3_Data.zip archive available from the CE802 moodle page. In this part of
the assignment, you are asked to perform the following two tasks.
2
a) Investigate the performance of a number of machine learning procedures on this
dataset. Using the data in the file CE802_P3_Data.csv contained in the CE802_P3_Data.zip
archive, you are required to perform a comparative study of the following machine learning procedures:
❼ Linear Regression;
❼ at least two more ML technique to predict the value of the claim.
This company too uses Python internally and therefore Python with scikit-learn is the required
language and machine learning library for the problem. For this task, you are expected to submit a
Jupyter Notebook called CE802_P3_Notebook.ipynb containing the Python code used to perform
the comparative analysis and produce the results as well as the code used to perform the predictions
described in task “b” below.
b) Prediction on a hold-out test set. An additional dataset, CE802_P3_Test.csv, is provided
inside the CE802_P3_Data.zip archive. Target values are withheld for this test set (i.e. the “Value”
column is empty). In this second task you are required to produce predictions of the records in
the test set using one approach of your choice among those tested in task “a” (for example the one
achieving the best performance). These data must not be used other than to test the algorithm
trained on the training data.
As part of your submission you should submit a new version of the file CE802_P3_Test.csv in
CSV format with the missing “Value” column replaced with the output predictions obtained using
the approach chosen. This second task will be marked based on the mean squared error on the test
set.
4. Report on the Investigation
After conducting the studies in parts 2 and 3, you are asked to write a report containing an account
of your investigation. There should be a brief summary of the experiments performed followed by
one or more tables and/or graphs summarizing the performance of the different solutions. Any
numerical data that you include should be in a suitable graphical or tabular form. The rest of the
report should concentrate on your interpretation of the results and what they tell you about the
relative strengths and weaknesses of the alternative methods when applied to the given data.
This document should consist of approximately 750–1500 words of narrative (i.e. excluding
references, pictures, and diagrams). Please report your word count on the title page. The document
must be submitted in PDF format with file name CE802_Report.pdf.
