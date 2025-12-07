#  -*- coding: utf-8 -*-
'''
from https://github.com/marcotcr/lime

Contains abstract functionality for learning locally linear sparse model.
'''
import numpy as np
import scipy as sp
import shap
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score



class XaiBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state) #岭回归
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_  # Parameters, or weights
            if sp.sparse.issparse(data): # True if x is a sparse array or matrix, False otherwise
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True) # Arrange the weights in descending order according to their absolute values
                return np.array([x[0] for x in feature_weights[:num_features]]) # According to the absolute value, take the index corresponding to the first num_features features and return them
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data, # The data generated by the feature corresponding mask is in the form of [[0,0,1,0,1,1,0,0,0,0,1,1,0,...],...]
                                   neighborhood_labels, # The corresponding label is in the form of [[0.6,0.4],[0,1],[0.8,0.2],...]
                                   distances, # Cosine distance
                                   label,
                                   num_features, #
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances) # Calculating weights
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data, # Nearby data
                                               labels_column, # The predicted value of the model for the corresponding category
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        easy_model = model_regressor

        easy_model.fit(neighborhood_data[1:, used_features], labels_column[1:], sample_weight=weights[1:])

        explainer = shap.Explainer(easy_model, neighborhood_data[1:, used_features])
        shap_values = explainer(neighborhood_data[0, used_features].reshape(1, -1)).values[0]












        pred_series = easy_model.predict(neighborhood_data[1:, used_features])

        MAE = mean_absolute_error(labels_column[1:], pred_series)  # 计算平均绝对误差（MAE）
        MSE = mean_squared_error(labels_column[1:], pred_series)  # 计算均方误差（MSE）
        RMSE = np.sqrt(MSE)  # 计算均方根误差（RMSE）
        EVS = explained_variance_score(labels_column[1:], pred_series)  # 解释方差分数（EVS）
        R2 = r2_score(labels_column[1:], pred_series)  # 计算决定系数（R-squared）
        Adjusted_R2 = 1 - (1 - R2) * (neighborhood_data[1:, used_features].shape[0] - 1) / (
                neighborhood_data[1:, used_features].shape[0] - easy_model.coef_.shape[
            0] - 1)  # 计算调整后的决定系数（Adjusted R-squared）

        MAE_weight = mean_absolute_error(labels_column[1:], pred_series, sample_weight=weights[1:])  # 计算平均绝对误差（MAE）
        MSE_weight = mean_squared_error(labels_column[1:], pred_series, sample_weight=weights[1:])  # 计算均方误差（MSE）
        RMSE_weight = np.sqrt(MSE_weight)  # 计算均方根误差（RMSE）
        EVS_weight = explained_variance_score(labels_column[1:], pred_series, sample_weight=weights[1:])  # 解释方差分数（EVS）
        R2_weight = r2_score(labels_column[1:], pred_series, sample_weight=weights[1:])  # 计算决定系数（R-squared）
        Adjusted_R2_weight = 1 - (1 - R2_weight) * (neighborhood_data[1:, used_features].shape[0] - 1) / (
                neighborhood_data[1:, used_features].shape[0] - easy_model.coef_.shape[
            0] - 1)  # 计算调整后的决定系数（Adjusted R-squared）


        # print(prediction_score)
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Linear Regressor fit metrics: ')
            print('  MAE', MAE)
            print('  MSE', MSE)
            print('  RMSE', RMSE)
            print('  EVS', EVS)
            print('  R^2', R2)  # Coefficient of determination
            print('  Adjusted R^2', Adjusted_R2)  # adjusted r-square

            print('  MAE_weight', MAE_weight)
            print('  MSE_weight', MSE_weight)
            print('  RMSE_weight', RMSE_weight)
            print('  EVS_weight', EVS_weight)
            print('  R^2_weight', R2_weight)  # Coefficient of determination
            print('  Adjusted R^2_weight', Adjusted_R2_weight)  # adjusted r-square

            print('The PD prob - Prediction_local: ', local_pred, )
            print('\n')

        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_), key=lambda x: np.abs(x[1]), reverse=True),
                sorted(zip(used_features, shap_values), key=lambda x: np.abs(x[1]), reverse=True),
                MAE,
                MAE_weight,
                MSE,
                MSE_weight,
                RMSE,
                RMSE_weight,
                EVS,
                EVS_weight,
                R2,
                R2_weight,
                Adjusted_R2,
                Adjusted_R2_weight,
                local_pred
                )





































