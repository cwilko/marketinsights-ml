import unittest
import os
import numpy as np
import pandas as pd
import marketinsights.utils.train as train
import quantutils.dataset.ml as mlutils
import quantutils.dataset.pipeline as ppl
from marketinsights.remote.ml import MIAssembly
from marketinsights.remote.models import MIModelServer
from marketinsights.model.ffnn import Perceptron
import tensorflow as tf

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

MODEL_ID = "3a491b1a-8af6-416d-aa14-f812cbd660bb"
DATASET_ID1 = "4234f0f1b6fcc17f6458696a6cdf5101"  # DOW

TRAINING_RUN = {
    "model_id": MODEL_ID,
    # TODO: This should be a dataset_desc, and not a list.
    "datasets": [DATASET_ID1]
}

#NUM_FEATURES = (2 * 4) + 1
#NUM_LABELS = 1
SEED = 42

TEST_MODEL_NAME = "testModel"
SHARED_RESULT = 0.5296804


class MLModelTestCase(unittest.TestCase):

    def setUp(self):

        #
        # Get dataset from MI API #
        #

        print("Loading data...")

        self.modelsvr = MIModelServer(
            serverConfigPath=root_dir + "../../config/modelserver/model_config.json",
            secret="marketinsights-k8s-cred")
        mi = MIAssembly(modelSvr=self.modelsvr, secret="marketinsights-k8s-cred")

        self.CONFIG = mi.get_model(MODEL_ID)
        TRN_CNF = self.CONFIG["training"]

        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        dataset, dataset_desc = mi.get_dataset_by_id(DATASET_ID1)
        dataset = dataset.astype(np.float32)

        NUM_FEATURES = dataset_desc["features"]
        self.NUM_LABELS = dataset_desc["labels"]

        # Crop training dates
        if "training_end_date" in TRN_CNF:
            dataset = dataset[TRN_CNF["training_start_date"]:TRN_CNF["training_end_date"]]

        self.TRAINING_SET_SIZE = int(TRN_CNF["training_window_size"] / 2)
        self.TEST_SET_SIZE = len(dataset) - self.TRAINING_SET_SIZE

        self.test_x, self.test_y = ppl.splitCol(dataset[self.TRAINING_SET_SIZE:], NUM_FEATURES)
        self.dataset = dataset

        TRAINING_RUN["id"] = MIAssembly.generateTrainingId(name=TEST_MODEL_NAME, dataset_desc=TRAINING_RUN["datasets"], model_id=TRAINING_RUN["model_id"])
        # Add Training run to DB
        mi.put_training_run(TRAINING_RUN)

        print(f"Creating model: {TRAINING_RUN['id']}")
        self.ffnn = Perceptron(TRAINING_RUN["id"], NUM_FEATURES, self.NUM_LABELS, self.CONFIG)

    def testFFNN_BootstrapTrain(self):

        ###############
        # Test Training
        ###############

        print("testFFNN_BootstrapTrain")

        TRN_CNF = self.CONFIG['training']
        print("Training", end='')

        metrics = train.bootstrapTrain(self.ffnn, self.dataset[:self.TRAINING_SET_SIZE], self.dataset[self.TRAINING_SET_SIZE:], iterations=1)
        # print(metrics)

        result = mlutils.evaluate(ppl.onehot(self.ffnn.predictions(self.test_x).numpy()), ppl.onehot(self.test_y), .0, display=False)

        print("".join(["Received : ", str(result)]))
        print(f"Expected : {str(SHARED_RESULT)}")
        self.assertTrue(np.allclose(result, SHARED_RESULT))  # Local results

        #########################
        # Test deploy and restore
        #########################

        # Deploy to model server
        self.modelsvr.deployModel(self.ffnn, version=1)

        # Restore variables state to model and check result
        self.modelsvr.restoreModel(model=self.ffnn, version=1)

        result = mlutils.evaluate(ppl.onehot(self.ffnn.predictions(self.test_x).numpy()), ppl.onehot(self.test_y), .0, display=False)
        self.assertTrue(np.allclose(result, SHARED_RESULT))  # Local results

        #####################################
        # Test prediction from deployed model
        #####################################

        # - Predict using REST API interface to model server
        # - Predict using gRPC interface to model server

        data = self.modelsvr.getPredictions(
            self.dataset.iloc[self.TRAINING_SET_SIZE:, :-self.NUM_LABELS],
            self.ffnn.modelName,
            debug=False
        )

        result = mlutils.evaluate(ppl.onehot(data[["y_pred0"]].values), ppl.onehot(self.test_y), .0, display=False)
        self.assertTrue(np.allclose(result, SHARED_RESULT))  # Local results
