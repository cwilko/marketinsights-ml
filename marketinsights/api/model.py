import tensorflow as tf
import importlib


class MarketInsightsModel(tf.Module):

    def __init__(self, name):
        super().__init__()
        self.modelName = name

    def save(self, root, version):
        return tf.saved_model.save(self, f"{root}/{self.modelName}/{version}", signatures=self.getSignatures())

    def restore(self, path):
        return tf.train.Checkpoint(self).restore(path).expect_partial()

    def getSignatures(self):
        signatures = self.__call__.get_concrete_function(tf.TensorSpec([None], tf.float32))

    @staticmethod
    def createInstance(modelClass, modelModule="marketinsights.model", opts={}):
        module = importlib.import_module(modelModule)
        modelInstance = getattr(module, modelClass)
        model = modelInstance(**opts)
        return model
