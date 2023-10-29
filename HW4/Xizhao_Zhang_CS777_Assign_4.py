from __future__ import print_function

import re
import sys
import numpy as np
from operator import add

from pyspark import SparkContext
numTopWords = 10000
def freqArray (listOfIndices):
	global numTopWords
	returnVal = np.zeros (numTopWords)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	mysum = np.sum (returnVal)
	returnVal = np.divide(returnVal, mysum)
	return returnVal


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: main_task1 <file> <file2> ", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="Assignment-4")

	### Task 1
	### Data Preparation
    
    corpus = sc.textFile(sys.argv[1])
	
    keyAndText = corpus.map(lambda x : (x[x.index('id="') + 
									   4 : x.index('" url=')], x[x.index('">') 
									  + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')

    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    allWords = keyAndListOfWords.flatMap(lambda x: x[1])
    wordCounts = allWords.countByValue()
    sortedWords = sorted(wordCounts.items(), key=lambda x: x[1], reverse=True)

    topWords = sc.parallelize(sortedWords[:10000])
    dictionary = topWords.zipWithIndex().map(lambda x: (x[0][0], x[1]))

    ### Include the following results in your report:
    print("Index for 'applicant' is",dictionary.filter(lambda x: x[0]=='applicant').take(1)[0][1])
    print("Index for 'and' is",dictionary.filter(lambda x: x[0]=='and').take(1)[0][1])
    print("Index for 'attack' is",dictionary.filter(lambda x: x[0]=='attack').take(1)[0][1])
    print("Index for 'protein' is",dictionary.filter(lambda x: x[0]=='protein').take(1)[0][1])
    print("Index for 'car' is",dictionary.filter(lambda x: x[0]=='car').take(1)[0][1])
    print("Index for 'in' is",dictionary.filter(lambda x: x[0]=='in').take(1)[0][1])




	### Task 2
	### Build your learning model
    dictionary_map = dictionary.collectAsMap()
    broadcast_dict = sc.broadcast(dictionary_map)



    #map words to indices
    def words_to_indices(words):
        global broadcast_dict
        return [broadcast_dict.value.get(word, -1) for word in words if broadcast_dict.value.get(word, -1) != -1]

    keyAndListOfIndices = keyAndListOfWords.mapValues(words_to_indices)
    keyAndFeatures = keyAndListOfIndices.mapValues(freqArray)
    def extract_label_from_key(key):
        if key.startswith("AU"):
            return 1
        else:
            return 0
    keyAndLabel = keyAndText.map(lambda x: (x[0], extract_label_from_key(x[0])))

    joined_rdd = keyAndLabel.join(keyAndFeatures)

    traindata = joined_rdd.map(lambda x: (x[1][0], x[1][1]))
    traindata.cache()
    train_size = traindata.count()

    # assign weights

    count_label_0 = traindata.filter(lambda x: x[0] == 0).count()
    count_label_1 = traindata.filter(lambda x: x[0] == 1).count()
    w0 = train_size/(2*count_label_0)
    w1 = train_size/(2*count_label_1)
    print(w0,w1)
    def LogisticRegression_weighted(traindata=traindata,
                        max_iteration = 20,
                        learningRate = 0.01,
                        regularization = 0.01,
                        mini_batch_size = 500000,
                        tolerance = 10e-8,
                        beta = 0.9,
                        beta2 = 0.999,
                        optimizer = 'SGD',  #optimizer: 'Momentum' / 'Adam' / 'Nesterov' / 'Adagrad' / 'RMSprop' / 'SGD' 
                        train_size=1
                        ):
            prev_cost = 0
            L_cost = []
            prev_validation = 0
            
            parameter_size = len(traindata.take(1)[0][1])
            np.random.seed(0)
            parameter_vector = np.random.normal(0, 0.1, parameter_size)
            momentum = np.zeros(parameter_size)
            prev_mom = np.zeros(parameter_size)
            second_mom = np.array(parameter_size)
            gti = np.zeros(parameter_size)
            epsilon = 10e-8
            
            for i in range(max_iteration):

                bc_weights = parameter_vector
                min_batch = traindata.sample(False, mini_batch_size / train_size, 1 + i)
        
                # treeAggregate(vector of gradients, total_cost, number_of_samples)
                # Calcualtion of positive class. Only the samples labeled as 1 are filtered and then  processed
                res1 = min_batch.filter(lambda x: x[0]==1).treeAggregate(
                    (np.zeros(parameter_size), 0, 0),
                    lambda x, y:(x[0]+\
                                (y[1])*(-y[0]+(1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
                                x[1]+\
                                y[0]*(-(np.dot(y[1], bc_weights)))+np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
                                x[2] + 1),
                    lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2])
                    )        
                # Calcualtion of negative class. Only the samples labeled as 0 are filtered and then processed
                res0 = min_batch.filter(lambda x: x[0]==0).treeAggregate(
                    (np.zeros(parameter_size), 0, 0),
                    lambda x, y:(x[0]+\
                                (y[1])*(-y[0]+(1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
                                x[1]+\
                                y[0]*(-(np.dot(y[1], bc_weights)))+np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
                                x[2] + 1),
                    lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2])
                    )        
                
                # The total gradients are a weighted sum
                gradients = w0*res0[0]+w1*res1[0]
                sum_cost = w0*res0[1]+w1*res1[1]
                num_samples = res0[2]+res1[2]
                
                cost =  sum_cost/num_samples + regularization * (np.square(parameter_vector).sum())

                # calculate gradients
                gradient_derivative = (1.0 / num_samples) * gradients + 2 * regularization * parameter_vector
                
                if optimizer == 'SGD':
                    parameter_vector = parameter_vector - learningRate * gradient_derivative

                if optimizer =='Momentum':
                    momentum = beta * momentum + learningRate * gradient_derivative
                    parameter_vector = parameter_vector - momentum
                    
                if optimizer == 'Nesterov':
                    parameter_temp = parameter_vector - beta * prev_mom
                    parameter_vector = parameter_temp - learningRate * gradient_derivative
                    prev_mom = momentum
                    momentum = beta * momentum + learningRate * gradient_derivative
                    
                if optimizer == 'Adam':
                    momentum = beta * momentum + (1 - beta) * gradient_derivative
                    second_mom = beta2 * second_mom + (1 - beta2) * (gradient_derivative**2)
                    momentum_ = momentum / (1 - beta**(i + 1))
                    second_mom_ = second_mom / (1 - beta2**(i + 1))
                    parameter_vector = parameter_vector - learningRate * momentum_ / (np.sqrt(second_mom_) + epsilon)

                if optimizer == 'Adagrad':
                    gti += gradient_derivative**2
                    adj_grad = gradient_derivative / (np.sqrt(gti)  + epsilon)
                    parameter_vector = parameter_vector - learningRate  * adj_grad
                
                if optimizer == 'RMSprop':
                    sq_grad = gradient_derivative**2
                    exp_grad = beta * gti / (i + 1) + (1 - beta) * sq_grad
                    parameter_vector = parameter_vector - learningRate / np.sqrt(exp_grad + epsilon) * gradient_derivative
                    gti += sq_grad
                    
                    
                print("Iteration No.", i, " Cost=", cost)
                
                # Stop if the cost is not descreasing
                if abs(cost - prev_cost) < tolerance:
                    print("cost - prev_cost: " + str(cost - prev_cost))
                    break
                prev_cost = cost
                L_cost.append(cost)

            return parameter_vector, L_cost
      
    
    # compare cost for different optimizer
    """ final_costs = {}
	optimizers = ['Momentum', 'Nesterov','Adam', 'Adagrad', 'RMSprop']
	for opt in optimizers:
		_, L_cost = LogisticRegression_optimized(
			traindata=traindata,
			max_iteration=5,
			learningRate=0.01,
			regularization=0.01,
			optimizer=opt
		)
		final_costs[opt] = L_cost[-1]  
	print(final_costs) """
	#{'Momentum': 1.6800396286880384, 'Nesterov': 1.6810817825571491, 
	# 'Adam': 1.683907388237019, 'Adagrad': 1.327466086259807, 'RMSprop': 1.0407297594364078}
	# Based on the results I may choose RMSprop as first choise

	### Print the top 5 words with the highest coefficients

    parameter_vector_RMSprop, L_cost_RMSprop = LogisticRegression_weighted(
						traindata=traindata,
						max_iteration = 20,
						learningRate = 0.01,
						regularization = 0.01,
						mini_batch_size = 512,
						optimizer = 'RMSprop'
                      )
	#find index for max abs for coef
    absindices = np.argsort(np.abs(parameter_vector_RMSprop))
    top_five_absindices = absindices[-5:][::-1]
    reversed_dictionary_map = {v: k for k, v in dictionary_map.items()}

    top_five = [reversed_dictionary_map.get(index, 'Unknown') for index in top_five_absindices]
    print(top_five)
    traindata.unpersist()


	### Task 3
	### Use your model to predict the category of each document
    test_corpus = sc.textFile(sys.argv[2])
    test_keyAndText = test_corpus.map(lambda x : (x[x.index('id="') + 
                                        4 : x.index('" url=')], x[x.index('">') 
                                        + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')

    test_keyAndListOfWords = test_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    test_keyAndListOfIndices = test_keyAndListOfWords.map(lambda x: (x[0], words_to_indices(x[1])))
    test_keyAndLabel = test_keyAndText.map(lambda x: (x[0], extract_label_from_key(x[0])))
    test_keyAndFeatures = test_keyAndListOfIndices.mapValues(freqArray)
    test_joined_rdd = test_keyAndLabel.join(test_keyAndFeatures)
    testdata = test_joined_rdd.map(lambda x: (x[1][0], x[1][1]))
    testdata.cache()
    test_num = testdata.count()

    # Create an RDD wiht the true value and the predicted value (true, predicted)
    predictions = testdata.map(lambda x: (x[0], 1 if np.dot(x[1],parameter_vector_RMSprop)>0 else 0))

    true_positive = predictions.map(lambda x: 1 if (x[0]== 1) and (x[1]==1) else 0).reduce(lambda x,y:x+y)
    false_positive = predictions.map(lambda x: 1 if (x[0]== 0) and (x[1]==1) else 0).reduce(lambda x,y:x+y)

    true_negative = predictions.map(lambda x: 1 if (x[0]== 0) and (x[1]==0) else 0).reduce(lambda x,y:x+y)
    false_negative = predictions.map(lambda x: 1 if (x[0]== 1) and (x[1]==0) else 0).reduce(lambda x,y:x+y)
    print("Performance Metrics: Logisitic Regression with Gradient Descent")
    # Print the Contingency matrix
    print("--Contingency matrix--")
    print(f" TP:{true_positive:6}  FP:{false_positive:6}")
    print(f" FN:{false_negative:6}  TN:{true_negative:6}")
    print("----------------------")

    # Calculate the Accuracy and the F1
    accuracy = (true_positive+true_negative)/(test_num)
    f1 = true_positive/(true_positive+0.5*(false_positive+false_negative))
    print(f"Accuracy = {accuracy}  \nF1 = {f1}")
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive != 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else 0

    print(f"Precision = {precision}")
    print(f"Recall = {recall}")

	

	
    sc.stop()