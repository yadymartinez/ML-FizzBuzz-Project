# ML FizzBuzz Project
Machine Learning solution for Fizz Buzz problem

## Fizz Buzz problem description 
Write a program that given the numbers from 1 to 100 print “None” for each number. 
But for multiples of the three print “Fizz” instead of “None” and for the multiples of five print  “Buzz”.
For numbers  which are multiples of both three an five print “FizzBuzz”.
    
There are many approaches to solving this problem. The most popular and well-known solution to this problem 
involves using conditional statements with a loop 
    
1. If the number (x) is divisible by three, the result must be “Fizz”
2. If the number (x) is divisible by five, the result must be “Buzz”1.
3. If the number (x) is divisible by both three and five, the result must be “FizzBuzz”
4. Else the result must be “None”

<image src="FizzBuzz.png" alt="Descripción de la imagen">

## FizzBuzz solution using Machine Learning approaches  

### Structure as a multi-class classification problem 

Fizzbuzz can be modeled as a multi-class classification problem.

**Input**: The most common option, is convert the number to its binary representation. 
           The binary representation can be fixed-length and each digit of the fixed-length 
           binary representation can be an input feature. 

**Target**: The target can be one of the four classes - fizz, buzz, fizzbuzz or none. 
            The model should predict which of the classes is most likely for an input number. 
            After the four classes are encoded and the model is built, it will return one of 
            four prediction labels. So we will also need a decoder function to convert the label 
            to the corresponding output.
       
**Model**: Logistic Regression (also called Logit Regression) is commonly used to estimate the probability that an 
           instance belongs to a particular class. 
           Logistic Regression is a classification algorithm used when the dependent (target) variables are categorical
           in nature- meaning the data can be grouped into discrete outputs ${0, 1, ..., k − 1}$.

Since we are dealing with categorical variables, logistical models must be used to map probabilities to predicted
labels of the data. 

There are three types of Logistic Regression:

1. Binomial: Where target variable is one of two classes
2. Multinomial: Where the target variable has three or more possible classes
3. Ordinal: Where the target variables have ordered categories

The Logistic Regression model can be generalized to support multiple classes directly, without having to train
and combine multiple binary classifiers. This is called Softmax Regression, or Multinomial Logistic Regression.
The idea is quite simple: when given an instance $x$, the Softmax Regression model first computes a score $s_k(x)$ for 
each class $k$, then estimates the probability of each class by applying the somax function (also called the normalized
exponential) to the scores. The equation to compute $s_k(x)$ should look familiar, as it is just like the equation for Linear
Regression prediction (see Equation 1).

$s_k(x) = \theta_k^T \cdot x$   Equation (1)

Just like the Logistic Regression classifier, the Softmax Regression classifier predicts the class with the
highest estimated probability (which is simply the class with the highest score), as shown in Equation 2.

$\hat{y}= max_k \theta(s(x))_k = \frac{exp(s_k(x))}{\sum_{j=1}^{K} exp(s_j(x))}$ Equation (2)

$\sigma(s(x))_k$ is the estimated probability that the instance $x$ belongs to class $k$ given the scores of each class for that instance.

The objective is to have a model that estimates a high probability for the target class (and consequently a low probability for the other
classes). Minimizing the cost function cross entropy. The gradient vector of this cost function with regards to $\theta_k$ is given by Equation 3:

Cross entropy gradient vector for class k

$\bigtriangledown \theta_k J(\Theta) = \frac{1}{m}\sum_{i=1}^{m}(\hat{p}_k^{(i)}-y_k^{(i)})x^{(i)} $ Equation (3)

Now you can compute the gradient vector for every class, then use Gradient Descent(or any other optimization algorithm) to find the parameter matrix $\Theta$ that minimizes
the cost function.|

              
### Generate FizzBuzz data

To generate the FizzBuzz data, the Dataset_Generator_ML_Fizz_Buzz(length_data, num_digits) function was defined: 

**Input**: -length_data: the total of integer that we use in the model.
-num_digits: the fixed-length binary representation can be 8, 10, 16, 32, 64. 

For example if length_data= 1024 we create the Input data as binary encoding from 1, 2, 3,...,1024 and num_digits is the length of the binary representation that will be encoding
the number. 

**Return**: X and y as a numpy array. 
-X contain the number encoding in binary representation. 
-y the labels fizz, buzz, fizzbuzz or none according to the number.  
                    
```python:
# Encoder the labels ('Fizz','FizzBuzz', 'Buzz', 'None') in (0,1,2,3)
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y.T) 
```  
Two additional helper function were defined:

```python:
# A boolean function return true or false if value is multiple of the 'multiple' value    
 def multiple(value, multiple):
    return True if value % multiple == 0 else False 
          
# Function return the encoding number in binary representation in length of num_digits
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])
```  
### Preprocessing data

StandardScaler() method calculates the mean and the standard deviation to use later for scaling the data. 
This method fits the parameters of the data and then transforms it. Standardizefeatures by removing the 
mean and scaling to unit variance.
       
``` python:
# Preprocessing the data
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```
    
### Build a Logistic-Regression model 

For uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. 
Implementation in python
```python:
# Logistic Regression Model for multi-class classification, l2-regularization
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs")
softmax_reg.fit(X, y)
```
    
### Report the accuracy score

The accuracy score was calculated with the test data.
```python:
score = softmax_reg.score(X_test, y_test)
print("Accuracy_LR_softmax:", score)
```
*The model report an: Accuracy_LR_softmax: 0.53*     

### Best perform on different classification algorithms using a ten fold Cross-Validation

 ```python:
 kFold = KFold(n_splits=10, random_state=42, shuffle=True)
 ```
 For each algorithm classification were calculated the accuracy score and save in a numpy by row:
 
 1. KNN– K- Nearest Neighbors Classifier (KNN)
 2. Random Forest (RF)
 3. Support Vector Machine (SVM)
 4. Stochastic Gradient Descent Classifier (SGDC)
    
 To report the best accuracy were calculated the maximum of the accuracy’s mean for each algorithm.
 
 The best models are Logistic Regression and Support Vector Machine with accuracy = 0.5351020408163266 

## FizzBuzz solution using Neural Network approaches
