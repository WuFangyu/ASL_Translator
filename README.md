## American Sign Languages (ASL) Classifier


An attempt to implement a more accurate learning model for classifier of ASL


Kaggle kernel: https://www.kaggle.com/wufangyu/kernel94c47f1083

DataSet: https://www.kaggle.com/grassknoted/asl-alphabet

Example:
Setup Virtual Env: 
```bash
$ mkdir -p ASLv1
$ virtualenv --python=python3.6 ASLv1
```

Activate the environment
```bash
$ source ASLv1/bin/activate
```

Install package 
```bash
$ pip install -r requirements.txt
```

Run
```bash
python example.py
```

Update:

After spending hours on improving the  CNN model and tweaking different hyperparameters,  the test data accuracy is still below 95%. Then I started investigating how to use other techniques such as ensemble learning (e.g. bagging,boosting,stacking ) to make better model 

Idea: Use soft voting from different machine learning models: CNN, SVM, RF(Random Forrest)…


TO-DO:
* Add suitable Local Feature Descriptors (e.g. SIFT, SURF) into SVM model
* reduce the size of HOG featurizer values to make all global features could fit in memory
* preprocess train-data to Improve the performance of K-Folder cross validation 


Updating...
