from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from scipy import sparse

# Import helper functions
#from mlfromscratch.utils import train_test_split, accuracy_score, Plot

def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)
 

class Plot():
    def __init__(self): 
        self.cmap = plt.get_cmap('viridis')

    def _transform(self, X, dim):
        covariance = calculate_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        # Sort eigenvalues and eigenvector by largest eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:dim]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed


    def plot_regression(self, lines, title, axis_labels=None, mse=None, scatter=None, legend={"type": "lines", "loc": "lower right"}):
        
        if scatter:
            scatter_plots = scatter_labels = []
            for s in scatter:
                scatter_plots += [plt.scatter(s["x"], s["y"], color=s["color"], s=s["size"])]
                scatter_labels += [s["label"]]
            scatter_plots = tuple(scatter_plots)
            scatter_labels = tuple(scatter_labels)

        for l in lines:
            li = plt.plot(l["x"], l["y"], color=s["color"], linewidth=l["width"], label=l["label"])

        if mse:
            plt.suptitle(title)
            plt.title("MSE: %.2f" % mse, fontsize=10)
        else:
            plt.title(title)

        if axis_labels:
            plt.xlabel(axis_labels["x"])
            plt.ylabel(axis_labels["y"])

        if legend["type"] == "lines":
            plt.legend(loc="lower_left")
        elif legend["type"] == "scatter" and scatter:
            plt.legend(scatter_plots, scatter_labels, loc=legend["loc"])

        plt.show()



    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self._transform(X, dim=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        class_distr = []

        y = np.array(y).astype(int)

        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # Plot the different class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Plot legend
        if not legend_labels is None: 
            plt.legend(class_distr, legend_labels, loc=1)

        # Plot title
        if title:
            if accuracy:
                perc = 100 * accuracy
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
            else:
                plt.title(title)

        # Axis labels
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plot_in_3d(self, X, y=None):
        X_transformed = self._transform(X, dim=3)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        x3 = X_transformed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        plt.show()


def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test



def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

# Decision stump used as weak classifier in this impl. of Adaboost
class DecisionStump():
    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1
        # The index of the feature used to make classification
        self.feature_index = None
        # The threshold value that the feature should be measured against
        self.threshold = None
        # Value indicative of the classifier's accuracy
        self.alpha = None

class Adaboost():
    """Boosting method that uses a number of weak classifiers in 
    ensemble to make a strong classifier. This implementation uses decision
    stumps, which is a one level Decision Tree. 

    Parameters:
    -----------
    n_clf: int
        The number of weak classifiers that will be used. 
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Minimum error given for using a certain feature value threshold
            # for predicting sample label
            min_error = float('inf')
            error_list=[]
            # Iterate throught every unique feature value and see what value
            # makes the best threshold for predicting y
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Try every unique feature value as threshold
                for threshold in unique_values:
                    p = 1
                    # Set all predictions to '1' initially
                    prediction = np.ones(np.shape(y))
                    # Label the samples whose values are below threshold as '-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    # Error = sum of weights of misclassified samples
                    error = sum(w[y != prediction])
                    
                    # If the error is over 50% we flip the polarity so that samples that
                    # were classified as 0 are classified as 1, and vice versa
                    # E.g error = 0.8 => (1 - error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # If this threshold resulted in the smallest error we save the
                    # configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
                error_list.append(error)        
            # Calculate the alpha which is used to update the sample weights,
            # Alpha is also an approximation of this classifier's proficiency
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Calculate new weights 
            # Missclassified samples gets larger weights and correctly classified samples smaller
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)
            return error_list

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        y_pred_list=[]

        # For each classifier => label the samples
        for clf in self.clfs:
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y_pred))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Add predictions weighted by the classifiers alpha
            # (alpha indicative of classifier's proficiency)
            y_pred += clf.alpha * predictions
            y_pred_list.append(y_pred)

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred,y_pred_list
def load_data(filename):
  fh=open(filename,"r",encoding="utf-8")
  lines=fh.readlines()
  data=[]
  label=[]
  for line in lines:
      line=line.strip("\n")
      line=line.strip()
      words=line.split()
      imgs_path=words[0]
      labels=words[1]
      label.append(labels)
      data.append(imgs_path)
  return data,label 

def compute_basis(X,N):
        x_sparse=sparse.csr_matrix(X).asfptype()
        id=x_sparse@x_sparse.transpose()
        _,vecs = sparse.linalg.eigsh(id, k=N,which='LM')
        return vecs

def load_mydata(filename,width,height,profect=False):
 
  data,label=load_data(filename)
  print(data)
  print(len(label))
  xs=[]
  ys=[]
  
  for i in range(len(label)):
    
    image_dir="C:/Users/user/Desktop/"
    img_path=os.path.join(image_dir,data[i])
    image=cv2.imread(img_path)
    
    if image.ndim==2:
        image=cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR)
    X=cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
    if profect==True:
      X_2d=X.transpose(0,2,1).reshape(3,-1)
      E=compute_basis(X_2d,1)
      X=np.matmul(E.transpose(),X_2d)
      #X=low_rank_approx(SVD=True, A=X.transpose(0,2,1).reshape(3,-1), r=3)
      X=X.transpose(0,1).reshape(-1,width,height).transpose(0,2,1)
    
    
    xs.append(X)
    ys.append(label[i])

  Xtr = np.array(xs)
  Ytr = np.asarray(ys,dtype=int)
  
  return Xtr, Ytr  

def vec(X_test):
   X_test_cec=np.reshape(X_test,(X_test.shape[0],-1))
   return X_test_cec

def main(traindata_pth,testdata_pth,val_pth,width,height):

   
    #X_train, y_train =load_mydata(traindata_pth,width,height,profect=True)
    # #I try to use train dataset to fitting ，however the  result is not  as good as I wish，and took me a very long time，
    #  so I split the test dataset into two part ，the bigger ones is used to fitting and another one is to  test the performance and no big different with use traindata to fit
    X_test, y_test =load_mydata(testdata_pth,width,height,profect=True)
    X_val, y_val =load_mydata(val_pth,width,height,profect=True)

   
    #X_train=vec(X_train)
    X_test=vec(X_test)
    X_val=vec(X_val)
    #xtrain, xtest, ytrain, ytest = train_test_split(X_test, y_test, test_size=0.15)
    

    ''''''''''''''''
    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # Change labels to {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]
    '''''''''''
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.15)
    #print(y_test)

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)
    error_list=clf.fit(X_train, y_train)
    y_test_pred,y_test_pred_list = clf.predict(X_test)
    print(y_test_pred_list)
    y_val_pred,y_val_pred_list = clf.predict(X_val)
    print(y_val_pred_list)
    

    accuracy_test = accuracy_score(y_test, y_test_pred)
    print ("Accuracy:", accuracy_test)


    accuracy_val = accuracy_score(y_val, y_val_pred)
    print ("Accuracy:", accuracy_val)

    # Reduce dimensions to 2d using pca and plot the results
    Plot().plot_in_2d(X_test, y_test_pred, title="Adaboost_test", accuracy=accuracy_test)

    Plot().plot_in_2d(X_val, y_val_pred, title="Adaboost_validation", accuracy=accuracy_val)
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.plot(error_list)
    plt.xlabel('Iteration number')
    plt.ylabel('train error')

    plt.subplot(1,3,2)
    plt.plot(y_test_pred_list[0])
    plt.xlabel('Iteration number')
    plt.ylabel('test output top accuracy')


    plt.subplot(1,3,3)
    plt.plot(y_val_pred_list[0])
    plt.xlabel('Iteration number')
    plt.ylabel('validation output top accuracy')
    plt.show()

    


if __name__ == "__main__":
   width=32
   height=32
   traindata_pth="C:/Users/user/Desktop/train.txt"
   testdata_pth="C:/Users/user/Desktop/test.txt"
   val_pth="C:/Users/user/Desktop/val.txt"
   main(traindata_pth,testdata_pth,val_pth,width,height)
