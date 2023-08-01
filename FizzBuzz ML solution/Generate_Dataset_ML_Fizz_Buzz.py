import numpy as np

def Dataset_Generator_ML_Fizz_Buzz(length_data,num_digits):

# A boolean function return true or false if value is multiple of the 'multiple' value
 def multiple(value, multiple):
    return True if value % multiple == 0 else False

# Function return the encoding number in binary representation in length of num_digits
 def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


# List of number in binary representation
 Data = []

# List of data class name
 labels = []

 i=1


 for i in range(1, length_data):

    if (multiple(i,3) & multiple(i,5)): # if the number is multiple of three and five
        #binary_encode(i, num_digits)
        Data.append(binary_encode(i, num_digits)) # Add to the list
        labels.append("FizzBuzz")
        print( str(i) + '_' + "FizzBuzz") # Print the number and word "FizzBuzz"
    elif multiple(i,3):   # if the number is only multiple of three
        Data.append(binary_encode(i, num_digits)) # Add to the list
        labels.append("Fizz")
        print(str(i) + '_' + "Fizz") # Print the number and word "Fizz"
    elif multiple(i, 5): # if the number is only multiple of three
        Data.append(binary_encode(i, num_digits)) # Add to the list
        labels.append("Buzz")
        print(str(i) + '_' + "Buzz") # Print the number and word "Buzz"
    else:  # if the sentences before isn't true
        Data.append(binary_encode(i, num_digits))  # Add to the list
        labels.append("None")
        print(str(i) + '_' + "None") # Print the number and word "None"

# Convert the lists in numpy array
 X = np.array(Data)
 y = np.array([labels])

 return X,y
