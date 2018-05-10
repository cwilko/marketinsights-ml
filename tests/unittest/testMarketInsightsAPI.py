import unittest
import pandas as pd
import numpy as np
import quantutils.dataset.pipeline as ppl
from quantutils.api.marketinsights import MarketInsights

class APITest(unittest.TestCase):

	def setUp(self):
		self.mi = MarketInsights('cred/MIOapi_cred.json')

	def test_predictions(self):
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