import numpy

def import_data(filename): 
    """ Takes a dataset and returns two arrays;
        One has the attributes for every patient X, the other has the corresponding class y
    """
    X = []
    y = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            for i in range(len(line)):
                if line[i] == '?':
                    line[i] = 'NaN'
            X += [line[:-2]]
            y += line[-2:-1]
            
    return X, y

def median(X):
    """ Finds median of the given dataset by columns
    """
    medians = []
    for i in range(len(X[0])):
        num = []
        for j in range(len(X)):
            if X[j][i] != 'NaN':
                num.append(float(X[j][i]))
        num.sort()
        medians.append(num[len(num)//2])
    
    return medians
        
def impute_missing(X):
    """ Imputes missing entries with the median of that feature
    """
    medians = median(X)
    for i in range(len(X[0])):
        for j in range(len(X)):
            if X[j][i] == 'NaN':
                X[j][i] = float(medians[i])
    return X


def discard_missing(X, y):
    """ Discards missing samples from the dataset
    """
    for i in range(len(X)):
        if "NaN" in X[i]:
            X.remove(X[i])
            y.remove(y[i])
    return X, y


def shuffle(X):
    """ Shuffles the order of the data entries in each row
    """
    for patient in X:
        numpy.random.shuffle(patient)
    
    return X

def meancalc(X):
    """ Helper function to find mean
    """
    mean = []
    for i in range(len(X[0])):
        sum = 0.0
        # finds the sum to calculate mean 
        for j in range(len(X)):
            sum += float(X[j][i])
        
        mean.append(sum/float(len(X)))
        
    return mean
            

def std_dev(X):
    """ Calculates the sample standard deviation of every feature
    """
    std_dev = []
    mean_vals = meancalc(X)

    for i in range(len(X[0])):
        res = 0
        # Calculates the sum of x - mean squared
        for j in range(len(X)):
            res += ((float(X[j][i]) - mean_vals[i])**2)
        
        std_dev.append((res/(len(X)-1)) ** (0.5))
    
    return std_dev


def remove_std_dev(X):
    """ Removes entries that are more than two standard deviations from the mean
    """
    std_vals = std_dev(X)
    mean_vals = meancalc(X)
    for i in range(len(X[0])):
        mean_val = mean_vals[i]
        std_val = std_vals[i]
        for j in range(len(X)):
            if X[j][i] < (mean_val - (2*std_val)):
                X.remove(X[j])
            elif X[j][i] > (mean_val + (2*std_val)):
                X.remove(X[j])
                
    return X

def standardize(X):
    """ standardizes all data points
    """
    std_vals = std_dev(X)
    mean_vals = meancalc(X)
    for i in range(len(X[0])):
        mean_val = mean_vals[i]
        std_val = std_vals[i]
        for j in range(len(X)):
            if std_val != 0:
                X[j][i] = (float(X[j][i])-mean_val)/std_val
            else:
                X[j][i] = (float(X[j][i])-mean_val)
    
    return X
        
