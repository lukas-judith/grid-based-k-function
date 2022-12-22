import numpy as np
import random
import os 
from time import time
from copy import deepcopy

from functional_data import *

import numpy as np
import random
import os 
from time import time
from copy import deepcopy

from functional_data import *


class SchillingTestFunctional():
    """
    Implements a two-sample test for functional data based on the approximated L2 norm. 
    Null distribution is approximated by shuffling and resampling from the the dataset.
    
    Reference: https://arxiv.org/abs/1610.06960
    """
    
    def __init__(self, data, k):
        self.data = data
        self.k = k
        # array holding the types of each function in the dataset
        self.Null_dist = None  
        # initialize nearest neighbor lists of all functions
        Function.get_nns_all_functions(data)
                    

    def compute_Schilling_statistic(self, data=None):
        """
        Computes Schilling's statistic for functional data.
        """
        if data is None:
            data = self.data
        sum_ = 0
        N = len(data)
        k = self.k

        for f1 in data:
            neighbor_types = np.array([f.type for f in f1.neighbors])
            same_types = (neighbor_types == f1.type)
            sum_ += np.sum(same_types[:k])
            
        schilling_stat = sum_ * 1/N * 1/k
        return schilling_stat
    
    
    def approximate_null_distribution(self, n_iter=10000, seed=None):
        """
        Approximate Null distribution by shuffling and resampling from the the dataset.
        """
        vals = []
        k = self.k
        t0 = time()

        for i in range(n_iter):
            if not seed is None:
                # use different seed for every shuffling
                seed_i = seed+i
                np.random.seed(seed_i)
            new_sample = self.shuffle_sample(self.data)
            s = self.compute_Schilling_statistic(new_sample)
            vals.append(s)    
        
        np.random.seed(None)
        
        t1 = time()
        diff_t = t1-t0

        print(f"Completed in {diff_t:.2f} seconds!")
        self.Null_dist = vals
        
    
    def get_critical_value(self, alpha):
        """
        Returns critical Null distribution value for a given significance level alpha.
        """
        if self.Null_dist is None:
            print("Error! Null distribution has not been approximated yet!")
            return
        return np.percentile(self.Null_dist, 100-alpha)
    

    def get_type_count(self, sample):
        """
        Counts different types in sample containing two types of functional data.
        """
        t1 = sample[0].type
        type1_count = 0
        type2_count = 0

        for f in sample:
            t = f.type
            if t == t1:
                type1_count += 1
            else:
                type2_count += 1

        return type1_count, type2_count
    
    
    def shuffle_sample(self, sample):
        """
        Shuffles sample containing two types of functional data.
        """
        type1_count, type2_count = self.get_type_count(sample)

        new_sample = deepcopy(sample)
        random.shuffle(new_sample)

        # randomly reorder the two samples into two new samples of the same sizes 
        for i in range(len(sample)):
            if i < type1_count:
                new_sample[i].type = 'type1'
            else:
                new_sample[i].type = 'type2'

        return new_sample
    
    
    def two_sample_test(self, alpha, bins=70, folder=".", save=False, show=True, printout=True):
        """
        Perform two-sample test using the Schilling's statistic for functional data.
        """
        k = self.k
        crit_value = self.get_critical_value(alpha)
        test_statistic = self.compute_Schilling_statistic()
        print(f"\nTesting at significance level {alpha}% with k={k}...")
        
        # adapt to different bin numbers etc.
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        
        ax.set_title("Approx. $H_0$ distr. and values of $T_{N,k}$")
        ax.set_xlabel("$T_{N,k}$")
        ax.set_ylabel('Density')
        
        # height of the vertical lines in the plot, ideally the max. height of the histogram
        max_height_hist = 20
        
        ax.hist(self.Null_dist, bins=bins, label='Null distribution', density=True, alpha=0.5)
        ax.vlines(x=test_statistic, ymin=0, ymax=max_height_hist, color='green', label='Observed value')
        ax.vlines(x=crit_value, ymin=0, ymax=max_height_hist, color='red', label='Critical value')
        ax.vlines(x=np.mean(self.Null_dist), ymin=0, ymax=max_height_hist, color='black', label='Mean', linestyle='dashed')
        ax.legend(loc='upper right')
        
        fig.text(.5, .05, f"Testing at significance level {alpha}%", ha='center')
        fig.text(.5, .0, f"Critical value is {crit_value:.4f}, test statistic has value {test_statistic:.4f}", ha='center')
        
        if printout:
            print(f"Critical value is {crit_value:.4f}, test statistic has value {test_statistic:.4f}")
            if test_statistic>crit_value:
                print("H_0 rejected")
            else:
                print("H_0 not rejected")
        
        if test_statistic>crit_value:
            fig.text(.5, -.05, f"Null hypothesis is rejected at significance level {alpha}%", ha='center')
        else:
            fig.text(.5, -.05, f"Null hypothesis cannot be rejected at significance level {alpha}%", ha='center')
   
        plt.title("Approx. $H_0$ distr. and values of $T_{N,k}$"+f", k={k}")
        if save:
            plt.savefig(os.path.join(folder, f"two_sample_test_k{k}_alpha{str(alpha).replace('.','')}.pdf"), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()