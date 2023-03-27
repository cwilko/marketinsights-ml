import numpy as np
import pandas as pd
import quantutils.dataset.pipeline as ppl


def sample(training_set, method="RANDOM", prop=.9, loo=0, boost=[]):
    if (method == "RANDOM"):
        training_set = training_set.sample(frac=1).reset_index(drop=True)
        idx = np.arange(0, len(training_set)) / float(len(training_set))
        return [training_set[idx < prop], training_set[idx >= prop]]
    elif (method == "LOO"):
        idx = np.array(range(0, len(training_set)))
        return [training_set[idx != loo], training_set[idx == loo]]
    elif (method == "BOOTSTRAP"):
        idx = np.array(range(0, len(training_set)))
        sample = np.random.choice(idx, len(training_set), replace=True)
        return pd.DataFrame(training_set.values[sample, :]), training_set[~np.in1d(idx, sample)]
    elif (method == "BOOSTING"):
        idx = np.array(range(0, len(training_set)))
        sample = np.random.choice(idx, len(training_set), replace=True, p=boost)
        return pd.DataFrame(training_set.values[sample, :]), training_set[~np.in1d(idx, sample)]


def bootstrapTrain(model, training_set, test_set, iterations):

    metrics = {
        "train_loss": [],
        "train_precision": [],
        "val_loss": [],
        "val_precision": [],
        "test_loss": [],
        "test_precision": [],
        "test_predictions": []
    }

    NUM_FEATURES = model.NUM_FEATURES
    train_X, train_y = ppl.splitCol(training_set, NUM_FEATURES)
    test_X, test_y = ppl.splitCol(test_set, NUM_FEATURES)

    for i in range(0, iterations):

        print(".", end='')

        # TODO: If we want to bootstrap a new model each time then reset here.
        # This would give us an ensemble of models rather than using the bootstrap
        # to influence the next batch.
        # model.resetVariablesToRandom()

        train_sample, val_sample = sample(training_set, method="BOOTSTRAP", loo=i)

        train_sample_X, train_sample_y = ppl.splitCol(train_sample, NUM_FEATURES)
        val_sample_X, val_sample_y = ppl.splitCol(val_sample, NUM_FEATURES)

        model.train(train_sample_X, train_sample_y, epochs=1)

        #metrics["train_loss"].append(model.loss(model(train_X), train_y))
        # metrics["train_precision"].append(results["train_precision"]["mean"])
        # metrics["val_loss"].append(results["val_loss"]["mean"])
        # metrics["val_precision"].append(results["val_precision"]["mean"])
        #metrics["test_loss"].append(model.loss(model(test_X), test_y))
        # metrics["test_precision"].append(results["test_precision"]["mean"])

    # For ensembles
    # results = {
        #"train_loss": {"mean": np.nanmean(metrics["train_loss"]), "std": np.nanstd(metrics["train_loss"]), "values": metrics["train_loss"]},
        #"train_precision": {"mean": np.nanmean(metrics["train_precision"]), "std": np.nanstd(metrics["train_precision"]), "values": metrics["train_precision"]},
        #"val_loss": {"mean": np.nanmean(metrics["val_loss"]), "std": np.nanstd(metrics["val_loss"]), "values": metrics["val_loss"]},
        #"val_precision": {"mean": np.nanmean(metrics["val_precision"]), "std": np.nanstd(metrics["val_precision"]), "values": metrics["val_precision"]},
        #"test_loss": {"mean": np.nanmean(metrics["test_loss"]), "std": np.nanstd(metrics["test_loss"]), "values": metrics["test_loss"]},
        #"test_precision": {"mean": np.nanmean(metrics["test_precision"]), "std": np.nanstd(metrics["test_precision"]), "values": metrics["test_precision"]},
        #"test_predictions": metrics["test_predictions"],
        #"weights": metrics["weights"],
    #}

    return metrics

###
# BOOSTING
###


def boostingTrain(model, training_set, test_set, lamda, iterations, debug=False):

    metrics = {
        "train_loss": [],
        "train_precision": [],
        "val_loss": [],
        "val_precision": [],
        "test_loss": [],
        "test_precision": [],
        "test_predictions": [],
        "weights": []
    }

    NUM_FEATURES = model.featureCount()
    test_X, test_y = ppl.splitCol(test_set, NUM_FEATURES)
    train_X, train_y = ppl.splitCol(training_set, NUM_FEATURES)
    threshold = 0  # For boosting to work this must be 0
    boost = np.array([1.0 / len(training_set)] * len(training_set))

    for i in range(0, iterations):

        print(".", end='')

        train_sample, val_sample = sample(training_set, method="BOOSTING", boost=boost)

        train_sample_X, train_sample_y = ppl.splitCol(train_sample, NUM_FEATURES)
        val_sample_X, val_sample_y = ppl.splitCol(val_sample, NUM_FEATURES)

        results = model.train(
            {'features': train_sample_X, 'labels': train_sample_y, 'lamda': lamda},
            {'features': val_sample_X, 'labels': val_sample_y, 'lamda': lamda},
            {'features': test_X, 'labels': test_y, 'lamda': lamda},
            threshold, 1, debug)

        # Evaluate the results and calculate the odds of misclassification
        _, _, train_predictions = model.evaluate(model.to_feed_dict({'features': train_X, 'labels': train_y, 'lamda': lamda}), threshold)
        precision = np.argmax(ppl.onehot(train_predictions), axis=1) == np.argmax(ppl.onehot(train_y), axis=1)  # TODO : This only works for onehot encoding
        epsilon = sum(boost[~precision])
        delta = epsilon / (1.0 - epsilon)
        boost[precision] = boost[precision] * delta
        boost = boost / sum(boost)

        metrics["train_loss"].append(results["train_loss"]["mean"])
        metrics["train_precision"].append(results["train_precision"]["mean"])
        metrics["val_loss"].append(results["val_loss"]["mean"])
        metrics["val_precision"].append(results["val_precision"]["mean"])
        metrics["test_loss"].append(results["test_loss"]["mean"])
        metrics["test_precision"].append(results["test_precision"]["mean"])
        metrics["test_predictions"].append(results["test_predictions"])
        metrics["weights"].append(results["weights"][0])  # Because we called train() with only 1 iteration

    results = {
        "train_loss": {"mean": np.nanmean(metrics["train_loss"]), "std": np.nanstd(metrics["train_loss"]), "values": metrics["train_loss"]},
        "train_precision": {"mean": np.nanmean(metrics["train_precision"]), "std": np.nanstd(metrics["train_precision"]), "values": metrics["train_precision"]},
        "val_loss": {"mean": np.nanmean(metrics["val_loss"]), "std": np.nanstd(metrics["val_loss"]), "values": metrics["val_loss"]},
        "val_precision": {"mean": np.nanmean(metrics["val_precision"]), "std": np.nanstd(metrics["val_precision"]), "values": metrics["val_precision"]},
        "test_loss": {"mean": np.nanmean(metrics["test_loss"]), "std": np.nanstd(metrics["test_loss"]), "values": metrics["test_loss"]},
        "test_precision": {"mean": np.nanmean(metrics["test_precision"]), "std": np.nanstd(metrics["test_precision"]), "values": metrics["test_precision"]},
        "test_predictions": metrics["test_predictions"],
        "weights": metrics["weights"]
    }

    if debug:
        print("Iteration : %d Lambda : %.2f, Threshold : %.2f" % (i, lamda, threshold))
        print("Training loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" %
              (results["train_loss"]["mean"], results["train_loss"]["std"],
               results["train_precision"]["mean"], results["train_precision"]["std"]))
        print("Validation loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" %
              (results["val_loss"]["mean"], results["val_loss"]["std"],
               results["val_precision"]["mean"], results["val_precision"]["std"]))
        print("Test loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" %
              (results["test_loss"]["mean"], results["test_loss"]["std"],
               results["test_precision"]["mean"], results["test_precision"]["std"]))

    return results
