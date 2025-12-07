#  -*- coding: utf-8 -*-
'''
from https://github.com/marcotcr/lime

Functions for explaining classifiers that use Image data.
'''
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


from . import xai_base




class PointcloudExplanation(object):
    def __init__(self, raw_data, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.raw_data = raw_data
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}  # weight
        self.local_shap = {}

        self.mae = {}
        self.mae_weight = {}
        self.mse = {}
        self.mse_weight = {}
        self.rmse = {}
        self.rmse_weight = {}
        self.evs = {}
        self.evs_weight = {}
        self.r2 = {}
        self.r2_weight = {}
        self.adjusted_r2 = {}
        self.adjusted_r2_weight = {}
        self.local_pred = {}


    def get_weight_and_shap(self, label, method, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        if method == 'lime':
            exp = self.local_exp[label]
        elif method == 'shap':
            exp = self.local_shap[label]
        exp = dict(exp)
        mask = np.zeros(segments.shape, dtype=np.float64)
        for id in np.arange(len(segments)):
            mask[id] = exp[segments[id]]

        metrics = [self.mae[label], self.mae_weight[label],
                   self.mse[label], self.mse_weight[label],
                   self.rmse[label], self.rmse_weight[label],
                   self.evs[label], self.evs_weight[label],
                   self.r2[label], self.r2_weight[label],
                   self.adjusted_r2[label], self.adjusted_r2_weight[label]]

        return np.reshape(mask, (len(mask), -1)), metrics


class XaiPointcloudExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=True,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in xai_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = xai_base.XaiBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self,
                         raw_data, # 最原始的点云
                         y_ids, # 查看哪个特征对其的影响
                         classifier_fn,
                         segment_fn,
                         window_size,
                         stride_size,
                         model_name,
                         label_type,
                         device,
                         model,
                         interested_labels=[1], # 对哪个类别感兴趣，或者说想解释哪个类别
                         hide_color=None,
                         top_labels=None,
                         num_features=100000,
                         num_samples=1000, # 控制生成多少局部相邻数据
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see xai_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segments, fudged_data = segment_fn(raw_data)

        top = interested_labels

        # Generate nearby data and the corresponding model predicted labels
        data, labels = self.data_labels(raw_data, y_ids, fudged_data, segments,
                                        window_size, stride_size, model_name,
                                        classifier_fn, num_samples,
                                        label_type, device, model,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        # Cosine distance as sample weight
        distances = sklearn.metrics.pairwise_distances(data, data[0].reshape(1, -1), metric=distance_metric).ravel()

        ret_exp = PointcloudExplanation(raw_data, segments)

        if top_labels:
            top = np.argsort(labels[0])[-top_labels:] # argsort() descending sort
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse() # reverse Reverse the order of elements in a list

        for label in top:  # The label here determines which category to focus on
            (ret_exp.intercept[label],

             ret_exp.local_exp[label],
             ret_exp.local_shap[label],

             ret_exp.mae[label],
             ret_exp.mae_weight[label],

             ret_exp.mse[label],
             ret_exp.mse_weight[label],

             ret_exp.rmse[label],
             ret_exp.rmse_weight[label],

             ret_exp.evs[label],
             ret_exp.evs_weight[label],

             ret_exp.r2[label],
             ret_exp.r2_weight[label],

             ret_exp.adjusted_r2[label],
             ret_exp.adjusted_r2_weight[label],

             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(data, labels, distances, label, num_features, model_regressor=model_regressor, feature_selection=self.feature_selection)

        return ret_exp

    def data_labels(self,
                    raw_data, # pc
                    y_ids, # See which features affect the model
                    fudged_data,
                    segments, window_size, stride_size, model_name,
                    classifier_fn,
                    num_samples, # Controls how much local neighbor data is generated
                    label_type, device, model,
                    batch_size=1,
                    progress_bar=False):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0] # Segments are equivalent to the index corresponding to the superpixel blocks of the image.
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1 # Here it is set to 1, indicating that all superpixel blocks are displayed, representing the original input data
        # seqs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(raw_data)
            zeros = np.where(row == 0)[0] # Wherever it is equal to zero, take the average of that small part
            mask = np.zeros(segments.shape).astype(np.int32)
            for z in zeros:
                mask[segments == z] = 1 # The length of the row is the same as the number of different values in the segments, so here you can specify which small fragments (where equal to 1) are replaced with the mean
            x_ids = np.where(mask==1)[0]
            for y_id in y_ids:
                temp[x_ids, y_id] = fudged_data[x_ids, y_id] # Make sure to only replace the columns of interest (most importantly, make sure the stroke index value remains unchanged)

            preds = classifier_fn(temp, window_size, stride_size, model_name, label_type, device, model)
            labels.append(preds)


        return data, np.array(labels)
