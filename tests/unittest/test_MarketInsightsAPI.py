import unittest
import os
import json
import pandas as pd
import numpy as np
import quantutils.dataset.pipeline as ppl
import quantutils.model.utils as mlutils
from quantutils.api.auth import CredentialsStore
from quantutils.api.marketinsights import MarketInsights, Dataset
from quantutils.api.functions import Functions
from quantutils.api.assembly import MIAssembly
#from quantutils.model.mimodelclient import MIModelClient

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/"


class APITest(unittest.TestCase):

    def setUp(self):

        self.cred = CredentialsStore()
        self.mi = MarketInsights(self.cred)
        fun = Functions(self.cred)
        self.miassembly = MIAssembly(self.mi, fun)

    def testEndToEndPredictionFromDataset(self):

        TRAINING_RUN_ID = "94b227b9d7b22c920333aa36d23669c8"
        DATASET_ID = "4234f0f1b6fcc17f6458696a6cdf5101"

        #mc = MIModelClient(self.cred)
        #results = self.miassembly.get_local_predictions_with_dataset_id(mc, DATASET_ID, TRAINING_RUN_ID, start="2016-07-01", end="2016-07-15", debug=True)
        #results = pd.DataFrame(results["data"], results["index"])
        results = self.miassembly.get_predictions_with_dataset_id(DATASET_ID, TRAINING_RUN_ID, start="2016-07-01", end="2016-07-15", debug=True)
        results = mlutils.aggregatePredictions([results], "mean_all")

        '''
        # Results should look like this
		Date_Time
		2016-07-01 15:00:00-04:00  0.000000e+00
		2016-07-05 15:00:00-04:00  0.000000e+00
		2016-07-06 15:00:00-04:00  0.000000e+00
		2016-07-07 15:00:00-04:00  6.174025e-03
		2016-07-08 15:00:00-04:00  8.180070e-01
		2016-07-11 15:00:00-04:00  1.000000e+00
		2016-07-12 15:00:00-04:00  3.874419e-06
		2016-07-13 15:00:00-04:00  9.999999e-01
		2016-07-14 15:00:00-04:00  3.974110e-11
		2016-07-15 15:00:00-04:00  3.007612e-01
		'''

        self.assertEqual(np.nansum(results), 3.124945995554477)

    def testEndToEndPredictionFromRawData(self):

        TRAINING_RUN_ID = "94b227b9d7b22c920333aa36d23669c8"

        with open(root_dir + "data/testRawData.json") as data_file:
            testRawData = json.load(data_file)

        data = Dataset.jsontocsv(testRawData)
        data.columns = ["Open", "High", "Low", "Close"]

        results = self.miassembly.get_predictions_with_raw_data(data, TRAINING_RUN_ID)

        results = mlutils.aggregatePredictions([results], "mean_all")

        '''
		Date_Time
		2016-07-01 15:00:00-04:00  0.000000e+00
		2016-07-05 15:00:00-04:00  0.000000e+00
		2016-07-06 15:00:00-04:00  0.000000e+00
		2016-07-07 15:00:00-04:00  6.174025e-03
		2016-07-08 15:00:00-04:00  8.180070e-01
		2016-07-11 15:00:00-04:00  1.000000e+00
		2016-07-12 15:00:00-04:00  3.874419e-06
		2016-07-13 15:00:00-04:00  9.999999e-01
		2016-07-14 15:00:00-04:00  3.974110e-11
		2016-07-15 15:00:00-04:00  3.007612e-01
		'''

        self.assertEqual(np.nansum(results), 3.124945995554477)

    @DeprecationWarning
    def _test_predictions(self):
        predictions = pd.read_csv(root_dir + 'data/testPredictions.csv', index_col=0, parse_dates=True, header=None)

        # Clean up
        print("Cleaning up")
        resp = self.mi.delete_predictions("testMkt", "testModelId", debug=False)

        print("Posting predictions")
        resp = self.mi.put_predictions(predictions, "testMkt", "testModelId", debug=False)
        self.assertTrue('success' in resp)

        resp = self.mi.get_predictions("testMkt", "testModelId")
        self.assertTrue(predictions.index.equals(resp.index))
        self.assertTrue(np.allclose(predictions.values, resp.values))

        # Shuffle values and update stored predictions
        predictions2 = ppl.shuffle(predictions)
        predictions2.index = predictions.index
        predictions = predictions2

        print("Updating predictions")
        resp = self.mi.put_predictions(predictions, "testMkt", "testModelId", update=True)
        self.assertTrue('success' in resp)

        resp = self.mi.get_predictions("testMkt", "testModelId")
        self.assertTrue(predictions.index.equals(resp.index))
        self.assertTrue(np.allclose(predictions.values, resp.values))

        print("Cleaning up")
        resp = self.mi.delete_predictions("testMkt", "testModelId")

        resp = self.mi.get_predictions("testMkt", "testModelId")
        self.assertTrue(resp.empty)


if __name__ == '__main__':
    unittest.main()
