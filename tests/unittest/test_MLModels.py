from __future__ import print_function
import unittest

import numpy as np
import pandas as pd

import quantutils.dataset.pipeline as ppl
from quantutils.api.auth import CredentialsStore
from quantutils.api.marketinsights import MarketInsights
from quantutils.api.functions import Functions
from quantutils.api.assembly import MIAssembly
from quantutils.api.bluemix import CloudObjectStore
import quantutils.model.utils as mlutils
from quantutils.model.ml import Model

MODEL_ID = "3a491b1a-8af6-416d-aa14-f812cbd660bb"

DATASET_ID1 = "4234f0f1b6fcc17f6458696a6cdf5101" # DOW
DATASET_ID2 = "3231bbe5eb2ab84eb54c9b64a8dcea55" # SPY

TRAINING_RUN = {
        "model_id": MODEL_ID,
        "datasets": [
            DATASET_ID1,
            DATASET_ID2
        ]
    }

COS_BUCKET = "marketinsights-weights"
NUM_FEATURES = (2 * 4) + 1
NUM_LABELS = 1

cred = CredentialsStore()
cos = CloudObjectStore(cred)

class MLModelTestCase(unittest.TestCase):

	

	def setUp(self):

		#
		# Get dataset from MI API #
		#

		print("Loading data...")
		
		mi = MarketInsights(cred)
		fun = Functions(cred)
		self.miassembly = MIAssembly(mi, fun)

		TRAINING_RUN["id"] = cos.generateKey([str(TRAINING_RUN["datasets"]), str(TRAINING_RUN["model_id"])])
		
		mi.put_training_run(TRAINING_RUN)

		self.CONFIG = mi.get_model(MODEL_ID)	

		print("Creating model...")
		# Create ML model
		self.ffnn = Model(NUM_FEATURES, NUM_LABELS, self.CONFIG)		

		mkt1, mkt1_desc = mi.get_dataset_by_id(DATASET_ID1)
		mkt2, mkt2_desc = mi.get_dataset_by_id(DATASET_ID2)

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

		print("testFFNN_BootstrapTrain")

		TRN_CNF = self.CONFIG['training']
		print("Training", end='')
		
		results = mlutils.bootstrapTrain(self.ffnn, self.dataset[:self.TRAINING_SET_SIZE], self.dataset[self.TRAINING_SET_SIZE:], TRN_CNF['lamda'], TRN_CNF['iterations'], TRN_CNF['threshold'], True)
		predictions =  np.nanmean(results["test_predictions"], axis=0)
		result = mlutils.evaluate(ppl.onehot(predictions), ppl.onehot(self.test_y), .0)

		print("".join(["Received : ", str(result)]))
		print("Expected : (0.48139533, 1.0, 0.6499214935472659)")
		self.assertTrue(np.allclose(result, np.array([0.48139533, 1.0, 0.6499214935472659]))) # Local results

		##################
		# Test weights API
		##################

		# Save weights to Cloud Object Store
		newWeights = pd.DataFrame(results["weights"])
		newWeights.insert(0,'timestamp', [self.isect[self.TRAINING_SET_SIZE//2].value // 10**9] * len(newWeights))
		cos.put_csv(COS_BUCKET, TRAINING_RUN["id"], newWeights)

		loadedWeights = cos.get_csv(COS_BUCKET, TRAINING_RUN["id"])
		self.assertTrue(np.allclose(newWeights.values, loadedWeights.values))

		#####################################
		# Test prediction from loaded weights
		#####################################

		dataset = self.mkt1.iloc[:,:-NUM_LABELS][-50:]
		timestamps = dataset.index.astype(np.int64) // 10**9
		dataset = dataset.reset_index(drop=True)
		newPredictions = self.predict(timestamps, dataset, loadedWeights)
		self.assertTrue(np.allclose(newPredictions, results["test_predictions"][0][-100:][::2]))


		#####################################
		# Test prediction from Assembly
		#####################################
		dataset = self.mkt1.iloc[:,:-NUM_LABELS][-50:]
		assemblyPredictions = self.miassembly.get_predictions_with_dataset(dataset, TRAINING_RUN["id"])
		self.assertTrue(np.allclose(newPredictions.flatten(), assemblyPredictions[0].values.flatten(), rtol=1e-03))

	def testFFNN_BoostingTrain(self):

		###############
		# Test Training
		###############

		print("testFFNN_BoostingTrain")

		TRN_CNF = self.CONFIG['training']
		print("Training", end='')
		
		results = mlutils.boostingTrain(self.ffnn, self.dataset[:self.TRAINING_SET_SIZE], self.dataset[self.TRAINING_SET_SIZE:], TRN_CNF['lamda'], TRN_CNF['iterations'], True)
		predictions =  np.nanmean(results["test_predictions"], axis=0)
		result = mlutils.evaluate(ppl.onehot(predictions), ppl.onehot(self.test_y), .0)

		print("".join(["Received : ", str(result)]))
		print("Expected : (0.47674417, 1.0, 0.6456692811685794)")
		self.assertTrue(np.allclose(result, np.array([0.47674417, 1.0, 0.6456692811685794]))) # Local results

		##################
		# Test weights API
		##################

		# Save weights to Cloud Object Store
		newWeights = pd.DataFrame(results["weights"])
		newWeights.insert(0,'timestamp', [self.isect[self.TRAINING_SET_SIZE//2].value // 10**9] * len(newWeights))
		cos.put_csv(COS_BUCKET, TRAINING_RUN["id"], newWeights)

		loadedWeights = cos.get_csv(COS_BUCKET, TRAINING_RUN["id"])
		self.assertTrue(np.allclose(newWeights.values, loadedWeights.values))

		#####################################
		# Test prediction from loaded weights
		#####################################

		dataset = self.mkt2.iloc[:,:-NUM_LABELS][-50:]
		timestamps = dataset.index.astype(np.int64) // 10**9
		dataset = dataset.reset_index(drop=True)
		newPredictions = self.predict(timestamps, dataset, loadedWeights)
		self.assertTrue(np.allclose(newPredictions, results["test_predictions"][0][-100:][1::2]))

		#####################################
		# Test prediction from Assembly
		#####################################
		dataset = self.mkt2.iloc[:,:-NUM_LABELS][-50:]
		assemblyPredictions = self.miassembly.get_predictions_with_dataset(dataset, TRAINING_RUN["id"])
		self.assertTrue(np.allclose(newPredictions.flatten(), assemblyPredictions[0].values.flatten(), rtol=1e-03))

	# Function to take dates, dataset info for those dates
	def predict(self, timestamps, dataset, weights=None):

	    # Load timestamps from weights db (or load all weights data)
	    if (weights is None):
	    	weights = cos.get_csv(COS_BUCKET, TRAINING_RUN["id"])
	    wPeriods = weights["timestamp"].values

	    # x = for each dataset timestamp, match latest available weight timestamp
	    latestPeriods = np.zeros(len(timestamps)) 
	    uniqueWPeriods = np.unique(wPeriods)  # q
	    mask = timestamps>=np.min(uniqueWPeriods)
	    latestPeriods[mask] = [uniqueWPeriods[uniqueWPeriods<=s][-1] for s in timestamps[mask]]

	    # for each non-duplicate timestamp in x, load weights into model for that timestamp
	    results = np.empty((len(dataset), NUM_LABELS))
	    for x in np.unique(latestPeriods):
	        # run dataset entries matching that timestamp through model, save results against original timestamps
	        mask = latestPeriods==x
	        results[mask] = np.nanmean(self.ffnn.predict(weights[wPeriods==x].values[:,1:], dataset[mask]), axis=0)
	    
	    return results    

if __name__ == '__main__':
    unittest.main()