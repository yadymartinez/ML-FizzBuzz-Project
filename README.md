<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#ML-FizzBuzz-Project">ML-FizzBuzz-Project</a>
      <ul>
        <li><a href="#Fizz Buzz Problem Description">Fizz Buzz Problem Description</a></li>
      </ul>
    </li>
    <li>
      <a href="#FizzBuzz solution using Machine Learning approaches with Logistic Regression algorithm">FizzBuzz solution using Machine Learning approaches with Logistic Regression algorithm</a>
      <ul>
        <li><a href="#Structure the problem as a multi-class classification problema">Structure the problem as a multi-class classification problema </a></li>
        <li><a href="#Generate the fizzbuzz data ">Generate the fizzbuzz data</a></li>      
        <li><a href="#Divide the data into training and test set">Divide the data into training and test set</a></li> 
        <li><a href="#Preprocessing data">Preprocessing data</a></li>    
        <li><a href="#Build a logistic regression model form sklearn library">Build a logistic regression model form sklearn library</a></li>    
        <li><a href="#Report the accuracy score with test data (1-100)">Report the accuracy score with test data (1-100)</a></li>    
        <li><a href="#Select the best perform on different classification algorithms using a ten fold-cross validation">Select the best perform on different classification algorithms using a ten fold-cross validation</a></li>    
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
c 
<!-- ML-FizzBuzz-Project -->
## ML-FizzBuzz-Project
Machine Learning solution for Fizz Buzz problem

<!-- Fizz Buzz Problem Description -->
## Fizz Buzz Problem Description 
    Write a program that given the numbers from 1 to 100 print “None” for each number. But for multiples of the three print “Fizz” instead of “None” and for the multiples of five print  “Buzz”.       For numbers  which are multiples of both three an five print “FizzBuzz”.
    There are many approaches to solving this problem. The most popular and well-known solution to this problem involves using conditional statements with a loop 
        1. If the number (x) is divisible by three, the result must be “Fizz”
        2. If the number (x) is divisible by five, the result must be “Buzz”
        3. If the number (x) is divisible by both three and five, the result must be “FizzBuzz” 
        4. Else the result must be “None”

<!-- ML-FizzBuzz-Project -->
## FizzBuzz solution using Machine Learning approaches with Logistic Regression algorithm   

## Structure the problem as a multi-class classification problem 
      Fizzbuzz can be modeled as a multi-class classification problem.
       Input: The most common option, is convert the number to its binary representation. The binary representation can be fixed-length and each digit of the fixed-length binary representation                  can be an input feature. 

        Target: The target can be one of the four classes - fizz, buzz, fizzbuzz or none. The model should predict which of the classes is most likely for an input number. After the four classes                    are encoded and the model is built, it will return one of four prediction labels. So we will also need a decoder function to convert the label to the corresponding output.

        Imagen under construction

        Model: Logistic Regression (also called Logit Regression) is commonly used to estimate the probability that an instance belongs to a particular class. Logistic Regression is a                              classification algorithm used when the dependent (target) variables are categorical in nature- meaning the data can be grouped into discrete outputs {0, 1, ..., k − 1}.
               Since we are dealing with categorical variables, logistical models must be used to map probabilities to predicted labels of the data. 

               There are three types of Logistic Regression:
                    1) Binomial: Where target variable is one of two classes
                    2) Multinomial: Where the target variable has three or more possible classes
                    3) Ordinal: Where the target variables have ordered categories

              The Logistic Regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers. This is called Softmax                        Regression, or Multinomial Logistic Regression.
              
  ## Generate the fizzbuzz data         
       To generate the FizzBuzz data, the Dataset_Generator_ML_Fizz_Buzz(length_data, num_digits) function was defined: 
           Input : length_data: the total of integer that we use in the model. 
           num_digits: the fixed-length binary representation can be 8, 10, 16, 32, 64. 

        For example if length_data= 1024 we create the Input data as binary encoding from 1, 2, 3, …. until 1024 and num_digits is the length of the binary representation that will be encoding
        the number. 

            Return: X and y as a numpy array. 
                    X contain the number encoding in binary representation. 
                    y the labels fizz, buzz, fizzbuzz or none according to the number.  

                   # Encoder the labels ('Fizz','FizzBuzz', 'Buzz', 'None') in (0,1,2,3)
                    label_encoder = preprocessing.LabelEncoder()
                     y = label_encoder.fit_transform(y.T)

          Two additional helper function were defined:

          # A boolean function return true or false if value is multiple of the 'multiple' value
           def multiple(value, multiple):
                return True if value % multiple == 0 else False 

          # Function return the encoding number in binary representation in length of num_digits
            def binary_encode(i, num_digits):
            return np.array([i >> d & 1 for d in range(num_digits)])
            
## Divide the data into training and test set
      The data was splited in training and test sets. 
       Training set contained all the value with index start in 100 and end = length_data.  
       Test set we use the first 100 number.
        # Dividing X, y into training and test data, we use the first 100 number for test set
        X_train = X[100:]
        y_train = y[100:]
        X_test = X[:100]
        y_test = y[:100]

  ## Preprocessing data
       StandardScaler() method calculates the mean and the standard deviation to use later for scaling the data. This method fits the parameters of the data and then transforms it. Standardize             features by removing the mean and scaling to unit variance.
        # Preprocessing the data
        sc = preprocessing.StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
    
  ## Build a logistic regression model form sklearn library 
       For uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. And 
       Implementation in python
       # Logistic Regression Model for multi-class classification, l2-regularization
       softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs")
       softmax_reg.fit(X, y)
    
  ## Report the accuracy score with test data (1-100)
       The accuracy score was calculated with the test data.
        score = softmax_reg.score(X_test, y_test)
        print("Accuracy_LR_softmax:", score)
        The model report an: Accuracy_LR_softmax: 0.53 %
        
  ## Select the best perform on different classification algorithms using a ten fold-cross validation
       
       kFold = KFold(n_splits=10, random_state=42, shuffle=True)
       For each algorithm classification were calculated the accuracy score and save in a numpy by row. 
       KNN – K- Nearest Neighbors Classifier (KNN)
       Random Forest -(RF)
       Support Vector Machine (SVM)
       Stochastic Gradient Descent Classifier (SGDC)
       To report the best accuracy were calculated the maximum of the accuracy’s mean for each algorithm.
       The best models are Logistic Regression and Support Vector Machine with accuracy = 0.5351020408163266 
 
# FizzBuzz solution using Neural Network approaches
