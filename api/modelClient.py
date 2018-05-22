import numpy as np
import pandas as pd
import pytz

from quantutils.api.bluemix import CloudObjectStore
from quantutils.api.marketinsights import MarketInsights, Dataset
from quantutils.model.ml import Model

COS_BUCKET = "marketinsights-weights"

cos = CloudObjectStore('api/cred/ibm_cos_cred.json')
mi = MarketInsights('api/cred/MIOapi_cred.json')

# TODO : Pull out of pipeline config?
##### Specific to the data ##
NUM_FEATURES = (2 * 4) + 1
NUM_LABELS = 1
#############################

class MIModelClient():

	models = {}

	def score(self, model_id, model_key, dataset):
		model = self.getModelInstance(model_id)
		weights = cos.get_csv(COS_BUCKET, model_key)
		index = pd.DatetimeIndex(dataset["index"], tz=pytz.timezone(dataset["tz"]))
		predictions = self.getPredictions(model, index.astype(np.int64) // 10**9, np.array(dataset["data"]), weights) 
		return Dataset.csvtojson(pd.DataFrame(predictions, index), dataset["market"], model_id)

	def getModelInstance(self, model_id):
		if (model_id not in self.models.keys()):
			self.models[model_id] = self.createModelInstance(model_id)
		return self.models[model_id]

	def createModelInstance(self, model_id):
		model_config = mi.get_model(model_id)
		# Create ML model
		return Model(NUM_FEATURES, NUM_LABELS, model_config)

	# Function to take dates, dataset info for those dates
	def getPredictions(self, model, timestamps, dataset, weights=None):

	    # Load timestamps from weights db (or load all weights data)
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
	        results[mask] = np.nanmean(model.predict(weights[wPeriods==x].values[:,1:], dataset[mask]), axis=0)
	    
	    return results    