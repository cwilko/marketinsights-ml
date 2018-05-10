from __future__ import print_function
import unittest

import numpy as np
import pandas as pd

import quantutils.dataset.pipeline as ppl
from quantutils.api.marketinsights import MarketInsights
from quantutils.api.bluemix import CloudObjectStore
import quantutils.model.utils as mlutils
from quantutils.model.ml import Model

class MLModelTestCase(unittest.TestCase):

	def setUp(self):

		MODEL_ID = "3a491b1a-8af6-416d-aa14-f812cbd660bb"

		MARKET1 = "DOW"
		MARKET2 = "SPY"

		PIPELINE_ID = "marketdirection"

		self.MODEL_KEY = "".join([MODEL_ID, "_", PIPELINE_ID, "_", MARKET1, "_", MARKET2, "-gz.csv"])
		self.COS_BUCKET = "marketinsights-weights"

		#
		# Get dataset from MI API #
		#

		print("Loading data...")
		mi = MarketInsights('cred/MIOapi_cred.json')
		self.cos = CloudObjectStore('cred/ibm_cos_cred.json')

		self.CONFIG = mi.get_model(MODEL_ID)

		NUM_FEATURES = (2 * 4) + 1
		NUM_LABELS = 2
		

		print("Creating model...")
		# Create ML model
		self.ffnn = Model(NUM_FEATURES, NUM_LABELS, self.CONFIG)		

		mkt1 = mi.get_dataset(MARKET1, PIPELINE_ID)
		mkt2 = mi.get_dataset(MARKET2, PIPELINE_ID)

		# Interleave (part of the "added insight" for this model)
		self.mkt1, self.mkt2, self.isect = ppl.intersect(mkt1,mkt2)
		self.dataset = ppl.interleave(self.mkt1,self.mkt2)

		self.TEST_SET_SIZE = 430
		self.TRAINING_SET_SIZE = len(self.dataset) - self.TEST_SET_SIZE
		self.WINDOW_SIZE = self.TRAINING_SET_SIZE

		_ , self.test_y = ppl.splitCol(self.dataset[self.TRAINING_SET_SIZE:], NUM_FEATURES)

	def testFFNN_BootstrapTrain(self):

		###############
		# Test Training
		###############

		TRN_CNF = self.CONFIG['training']
		print("Training", end='')
		
		results = mlutils.bootstrapTrain(self.ffnn, self.dataset[:self.TRAINING_SET_SIZE], self.dataset[self.TRAINING_SET_SIZE:], TRN_CNF['lamda'], TRN_CNF['iterations'], TRN_CNF['threshold'], True)
		predictions =  np.nanmean(results["test_predictions"], axis=0)
		result = mlutils.evaluate(predictions, self.test_y, .0)

		print("".join(["Received : ", str(result)]))
		print("Expected : (0.5255814, 1.0, 0.6890244109701231)")
		self.assertTrue(np.allclose(result, np.array([0.5255814, 1.0, 0.6890244109701231]))) # Local results

		##################
		# Test weights API
		##################

		# Save weights to Cloud Object Store
		newWeights = pd.DataFrame(results["weights"])
		newWeights.insert(0,'timestamp', [self.isect[self.TRAINING_SET_SIZE//2].value // 10**9] * len(newWeights))
		self.cos.put_csv(self.COS_BUCKET, self.MODEL_KEY, newWeights)

		loadedWeights = self.cos.get_csv(self.COS_BUCKET, self.MODEL_KEY)
		self.assertTrue(np.allclose(newWeights.values, loadedWeights.values))

		#####################################
		# Test prediction from loaded weights
		#####################################

		dataset = self.mkt1.iloc[:,:-2][-50:]
		timestamps = dataset.index.astype(np.int64) // 10**9
		dataset = dataset.reset_index(drop=True)
		newPredictions = self.predict(timestamps, dataset, loadedWeights)
		self.assertTrue(np.allclose(newPredictions, results["test_predictions"][0][-100:][::2]))
		

	def testFFNN_BoostingTrain(self):

		###############
		# Test Training
		###############

		TRN_CNF = self.CONFIG['training']
		print("Training", end='')
		
		results = mlutils.boostingTrain(self.ffnn, self.dataset[:self.TRAINING_SET_SIZE], self.dataset[self.TRAINING_SET_SIZE:], TRN_CNF['lamda'], TRN_CNF['iterations'], True)
		predictions =  np.nanmean(results["test_predictions"], axis=0)
		result = mlutils.evaluate(predictions, self.test_y, .0)

		print("".join(["Received : ", str(result)]))
		print("Expected : (0.5232558, 1.0, 0.687022910321775)")
		self.assertTrue(np.allclose(result, np.array([0.5232558, 1.0, 0.687022910321775]))) # Local results

		##################
		# Test weights API
		##################

		# Save weights to Cloud Object Store
		newWeights = pd.DataFrame(results["weights"])
		newWeights.insert(0,'timestamp', [self.isect[self.TRAINING_SET_SIZE//2].value // 10**9] * len(newWeights))
		self.cos.put_csv(self.COS_BUCKET, self.MODEL_KEY, newWeights)

		loadedWeights = self.cos.get_csv(self.COS_BUCKET, self.MODEL_KEY)
		self.assertTrue(np.allclose(newWeights.values, loadedWeights.values))

		#####################################
		# Test prediction from loaded weights
		#####################################

		dataset = self.mkt2.iloc[:,:-2][-50:]
		timestamps = dataset.index.astype(np.int64) // 10**9
		dataset = dataset.reset_index(drop=True)
		newPredictions = self.predict(timestamps, dataset, loadedWeights)
		self.assertTrue(np.allclose(newPredictions, results["test_predictions"][0][-100:][1::2]))

	# Function to take dates, dataset info for those dates
	def predict(self, timestamps, dataset, weights=None):

	    # Load timestamps from weights db (or load all weights data)
	    if (weights is None):
	    	weights = self.cos.get_csv(self.COS_BUCKET, self.MODEL_KEY)
	    wPeriods = weights["timestamp"].values

	    # x = for each dataset timestamp, match latest available weight timestamp
	    latestPeriods = np.zeros(len(timestamps)) 
	    uniqueWPeriods = np.unique(wPeriods)  # q
	    mask = timestamps>=np.min(uniqueWPeriods)
	    latestPeriods[mask] = [uniqueWPeriods[uniqueWPeriods<=s][-1] for s in timestamps[mask]]

	    # for each non-duplicate timestamp in x, load weights into model for that timestamp
	    results = np.empty((len(dataset), 2))
	    for x in np.unique(latestPeriods):
	        # run dataset entries matching that timestamp through model, save results against original timestamps
	        mask = latestPeriods==x
	        results[mask] = np.nanmean(self.ffnn.predict(weights[wPeriods==x].values[:,1:], dataset[mask]), axis=0)
	    
	    return results    

if __name__ == '__main__':
    unittest.main()