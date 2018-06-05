import unittest
import json
import pandas as pd
import numpy as np
import quantutils.dataset.pipeline as ppl
from quantutils.api.marketinsights import MarketInsights, Dataset
from quantutils.api.functions import Functions
from quantutils.api.assembly import MIAssembly

class APITest(unittest.TestCase):

	def setUp(self):
		self.mi = MarketInsights('cred/MIOapi_cred.json')
		fun = Functions("cred/functions_cred.json")
		self.miassembly = MIAssembly(self.mi, fun)

	def testEndToEndPredictionFromDataset(self):

		TRAINING_RUN_ID = "ebc370105a5af4b3ef183445bc03a991"
		DATASET_ID = "265e2f7f3e06af1c6fc9e74434514c86"

		results = self.miassembly.get_predictions_with_dataset_id(DATASET_ID, TRAINING_RUN_ID, start="2016-07-01", end="2016-07-15")

		self.assertEqual(np.nansum(results), 4.399551914189942)

	def testEndToEndPredictionFromRawData(self):

		TRAINING_RUN_ID = "ebc370105a5af4b3ef183445bc03a991"

		with open("data/testRawData.json") as data_file:    
		    testRawData = json.load(data_file)

		data = Dataset.jsontocsv(testRawData)		
		data.columns = ["Open","High","Low","Close"]

		results = self.miassembly.get_predictions_with_raw_data(data, TRAINING_RUN_ID)

		self.assertEqual(np.nansum(results), 4.399551914189942)

	@DeprecationWarning
	def _test_predictions(self):
		predictions = pd.read_csv('data/testPredictions.csv', index_col=0, parse_dates=True, header=None)

		#Clean up
		print "Cleaning up"
		resp = self.mi.delete_predictions("testMkt", "testModelId", debug=False)

		print "Posting predictions"
		resp = self.mi.put_predictions(predictions, "testMkt", "testModelId", debug=False)
		self.assertTrue('success' in resp)

		resp = self.mi.get_predictions("testMkt", "testModelId")
		self.assertTrue(predictions.index.equals(resp.index))
		self.assertTrue(np.allclose(predictions.values, resp.values))

		# Shuffle values and update stored predictions
		predictions2 = ppl.shuffle(predictions)
		predictions2.index = predictions.index
		predictions = predictions2

		print "Updating predictions"
		resp = self.mi.put_predictions(predictions, "testMkt", "testModelId", update=True)
		self.assertTrue('success' in resp)

		resp = self.mi.get_predictions("testMkt", "testModelId")
		self.assertTrue(predictions.index.equals(resp.index))
		self.assertTrue(np.allclose(predictions.values, resp.values))

		print "Cleaning up"
		resp = self.mi.delete_predictions("testMkt", "testModelId")

		resp = self.mi.get_predictions("testMkt", "testModelId")
		self.assertTrue(resp.empty)


if __name__ == '__main__':
    unittest.main()