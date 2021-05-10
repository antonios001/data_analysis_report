# data_analysis_report
## Price Prediction using Linear Regression, Decision Trees and Random Forests

First, I downloaded the data file, titled *car.data*, from [https://code.datasciencedojo.com/datasciencedojo/datasets/tree/master/Car%20Evaluation].

I wrote a python script titled *multiunitregression.py* to process the data and apply 3 predictive models on the buying price, employing Linear Regression, Desision Trees and Random Forests. The *car.data* file came without headers, so I wrote function *save_file_with_headers* to add the headers described in the above url and convert the file to csv, having the *car.data.csv* file as output. I wrote function *ordinal_data_regression* to process the data and perform the prediction task. Given the fact that the attribute data were qualitative, but ordinal, I proceeded to encode each attribute column, from lowest to highest quality, starting from 0. Subsequently, I checked for attribute correlation, which could be useful in using stratified sampling based on the attribute most correlated to the buying price, to appropriately split the dataset to a training and testing set. I noticed that there is a slight negative correlation between buying price and car acceptability. Normally, I would choose to do stratified sampling based on correlated attributes, however, this correlation result seems counter-intuitive, so I just performed random sampling instead. 
     


