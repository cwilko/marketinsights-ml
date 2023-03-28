import unittest
import os
import json
import pandas as pd
import numpy as np
import quantutils.dataset.pipeline as ppl
import quantutils.dataset.ml as mlutils
from marketinsights.remote.ml import MIAssembly, Dataset
from marketinsights.remote.models import MIModelServer

#from quantutils.model.mimodelclient import MIModelClient

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
SHARED_RESULT = 4.594576955


class APITest(unittest.TestCase):

    def setUp(self):

        self.modelsvr = MIModelServer(secret="marketinsights-k8s-cred")
        self.assembly = MIAssembly(modelSvr=self.modelsvr, secret="marketinsights-k8s-cred")

    def testEndToEndPredictionFromDataset(self):

        TRAINING_RUN_ID = "testModel-4b8fcc0053f13d518c4056ba9e1e3cdc"
        DATASET_ID = "4234f0f1b6fcc17f6458696a6cdf5101"

        results = self.assembly.get_predictions_with_dataset_id(DATASET_ID, TRAINING_RUN_ID, start="2016-07-06", end="2016-07-15", debug=False)
        # print(results)
        results = np.nansum(results["y_pred0"].values)

        self.assertEqual(results, SHARED_RESULT)

    def testEndToEndPredictionFromRawData(self):

        TRAINING_RUN_ID = "testModel-4b8fcc0053f13d518c4056ba9e1e3cdc"

        with open(root_dir + "data/testRawData.json") as data_file:
            testRawData = json.load(data_file)

        data = Dataset.jsontocsv(testRawData)
        data.columns = ["Open", "High", "Low", "Close"]

        results = self.assembly.get_predictions_with_raw_data(data, TRAINING_RUN_ID, debug=False)
        results = np.nansum(results["y_pred0"].values)

        self.assertEqual(results, SHARED_RESULT)
