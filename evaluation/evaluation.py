
import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import gc

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
# Need to change this for different systems
# INCEPTION_V3_PATH = "/home/ens/AT74470/util_files/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "/export/livia/home/vision/Ymohammadi/util_files/classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"

# VGG_PATH = "/home/ens/AT74470/util_files/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
VGG_PATH = "/export/livia/home/vision/Ymohammadi/util_files/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG16 feature extraction model (excluding top layers)
vgg_model = VGG16(weights=VGG_PATH, include_top=False)
# Feature extraction model based on LPIPS-like metric (e.g., 'block5_conv3' layer)
lpips_feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_conv3').output)


# Add LPIPS score computation to runEvaluate function
def runEvaluate(ref_batch, sample_batch, FID=False, IS=False, sFID=False, prec_recall=False, KID=False, LPIPS=False, source_batch=None, intra_LPIPS=False, target_batch=None, verbose=True):
    """
    Evaluate several metrics, including LPIPS-like perceptual distance between datasets.
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    evaluator = Evaluator(sess)

    evaluator.warmup()

    if FID == True or IS == True or sFID==True or prec_recall==True or KID==True: 

        ref_acts = evaluator.read_activations(ref_batch)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch, ref_acts)

        sample_acts = evaluator.read_activations(sample_batch)
        sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_batch, sample_acts)

    results = {}

    if IS:
        IS_score = evaluator.compute_inception_score(sample_acts[0])
        results['IS'] = IS_score
    if FID:
        FID_score = sample_stats.frechet_distance(ref_stats)
        results['FID'] = FID_score
    if sFID:
        sFID_score = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        results['sFID'] = sFID_score
    if prec_recall:
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        results['Precision'] = prec
        results['Recall'] = recall
    if KID:
        KID_score = evaluator.compute_kid(ref_acts[0], sample_acts[0])
        results['KID'] = KID_score
    if LPIPS:
        lpips_score = compute_lpips_between_distributions(source_batch, sample_batch, lim=1000)
        results['LPIPS'] = lpips_score
    if intra_LPIPS:
        intra_lpips, intra_lpips_dict = compute_intra_cluster_feature_distance(target_batch, sample_batch, lim=1000)
        results['intra_LPIPS'] = intra_lpips
        results['intra_LPIPS_dict'] = intra_lpips_dict

    if verbose:
        for key, value in results.items():
            print(f"{key}: {value}")

    sess.close()
    K.clear_session()
    tf.reset_default_graph()
    gc.collect()
    tf.keras.backend.clear_session()  # For TensorFlow 2.x

    return results


# ___________________ LPIPS ______________________

def compute_lpips_between_distributions(npz_file1, npz_file2, lim=1000, batch_size=64):
    """
    Compute LPIPS-like score between two datasets, treating them as distributions.

    :param npz_file1: Path to first NPZ file.
    :param npz_file2: Path to second NPZ file.
    :param batch_size: Batch size for processing images.
    :return: Average LPIPS-like score between the two datasets.
    """
    with open_npz_array(npz_file1, 'arr_0') as reader1, open_npz_array(npz_file2, 'arr_0') as reader2:
        images1 = preprocess_images(np.concatenate([batch for batch in reader1.read_batches(batch_size)]))
        images2 = preprocess_images(np.concatenate([batch for batch in reader2.read_batches(batch_size)]))

        # Extract features from both datasets
        features1 = extract_features_from_vgg(images1, batch_size)
        features2 = extract_features_from_vgg(images2, batch_size)

        # Compute pairwise distances
        lpips_scores = []
        for i, j in zip(range(min(features1.shape[0], lim)), range(min(features2.shape[0], lim))):
            # dist = np.linalg.norm(features1[i] - features2[j])  # Euclidean distance
            dist = cosine(features1[i].flatten(), features2[i].flatten())
            lpips_scores.append(dist)

        # Compute average LPIPS score across all pairs
        avg_lpips_score = np.mean(lpips_scores)
        return avg_lpips_score

def extract_features_from_vgg(images, batch_size):
    """
    Extract deep features from VGG16 for LPIPS computation.

    :param images: Batch of input images.
    :param batch_size: Batch size for processing.
    :return: Extracted deep features.
    """
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_features = lpips_feature_extractor.predict(batch)
        features.append(batch_features)
    
    return np.concatenate(features, axis=0)

# Preprocess images for VGG16 model
def preprocess_images(images):
    images = images.astype(np.float32)
    images_resized = np.array([tf.image.resize(image, (224, 224)).numpy() for image in images])
    return preprocess_input(images_resized)



# ___________________ Intra-cluster LPIPS ______________________


# Compute intra-cluster feature distances (similar to intra-cluster LPIPS)
def compute_intra_cluster_feature_distance(npz_target, npz_generated, lim=1000, batch_size=64):
    with open_npz_array(npz_target, 'arr_0') as reader1, open_npz_array(npz_generated, 'arr_0') as reader2:
        images_target = preprocess_images(np.concatenate([batch for batch in reader1.read_batches(batch_size)]))
        images_generated = preprocess_images(np.concatenate([batch for batch in reader2.read_batches(batch_size)]))

        # Extract features from both datasets
        features_target = extract_features_from_vgg(images_target, batch_size)
        features_generated = extract_features_from_vgg(images_generated, batch_size)

        features_target = features_target[:10] # TODO TMP remove
        features_generated = features_generated[:lim]

        cluster_assignments = assign_to_clusters(features_generated, features_target)

        unique_clusters, frequencies = np.unique(cluster_assignments, return_counts=True)
        intra_lpips_dict = dict(zip(unique_clusters, frequencies))
        intra_cluster_distances = []

        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_assignments == cluster)[0]
            cluster_features = tf.gather(features_generated, cluster_indices)

            # Compute pairwise feature distances within the cluster
            pairwise_distances = []
            for i in range(len(cluster_features)):
                for j in range(i + 1, len(cluster_features)):
                    # dist = np.linalg.norm(cluster_features[i] - cluster_features[j])
                    dist = cosine(cluster_features[i].numpy().flatten(), cluster_features[j].numpy().flatten())
                    pairwise_distances.append(dist)

            # Average pairwise feature distance for this cluster
            if pairwise_distances:
                avg_distance = np.mean(pairwise_distances)
                intra_cluster_distances.append(avg_distance)

        # Average distances across all clusters
        return np.mean(intra_cluster_distances), intra_lpips_dict


# Assign generated images to clusters based on feature similarity to target images
def assign_to_clusters(generated_features, target_features):
    num_target_images = target_features.shape[0]
    num_generated_images = generated_features.shape[0]

    # Initialize cluster assignments
    cluster_assignments = np.zeros(num_generated_images)

    # Compute feature distances and assign each generated image to the closest target image (cluster)
    for i in range(num_generated_images):
        min_distance = float('inf')
        best_cluster = -1
        for j in range(num_target_images):
            # dist = np.linalg.norm(generated_features[i] - target_features[j])
            dist = cosine(generated_features[i].numpy().flatten(), target_features[j].numpy().flatten())
            if dist < min_distance:
                min_distance = dist
                best_cluster = j
        cluster_assignments[i] = best_cluster

    return cluster_assignments


# __________________________ FID _________________________________

class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                # raise ValueError("Imaginary component {}".format(m))
                return None
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(
        self,
        session,
        batch_size=64,
        softmax_batch_size=512,
    ):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.

        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                 dimension. The tuple is (pool_3, spatial).
        """
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(spatial_preds, axis=0),
        )

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
                obj["mu_s"], obj["sigma_s"]
            )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))


    def compute_kid(self, ref_activations: np.ndarray, sample_activations: np.ndarray) -> float:
        """
        Compute the Kernel Inception Distance (KID) between reference and sample activations.
        
        :param ref_activations: Features of real images.
        :param sample_activations: Features of generated images.
        :return: The KID score.
        """
        return compute_kid(ref_activations, sample_activations)

    def compute_prec_recall(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        session,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param session: the TensorFlow session.
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def warmup(self):
        feats, radii = (
            np.zeros([1, 2048], dtype=np.float32),
            np.zeros([1, 1], dtype=np.float32),
        )
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(self, features: np.ndarray, radii: np.ndarray, eval_features: np.ndarray):
        """
        Evaluate if new feature vectors are at the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= radii
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                radii[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0 : end1 - begin1, :], axis=1)

        return {
            "fraction": float(np.mean(batch_predictions)),
            "batch_predictions": batch_predictions,
            "max_realisim_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=np.bool_)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=np.bool_)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    """
    Calculate pairwise distances between vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self, session):
        self.session = session

        # Initialize TF graph to calculate pairwise distances.
        with session.graph.as_default():
            self._features_batch1 = tf.placeholder(tf.float32, shape=[None, None])
            self._features_batch2 = tf.placeholder(tf.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf.cast(self._features_batch1, tf.float16),
                tf.cast(self._features_batch2, tf.float16),
            )
            self.distance_block = tf.cond(
                tf.reduce_all(tf.math.is_finite(distance_block_16)),
                lambda: tf.cast(distance_block_16, tf.float32),
                lambda: _batch_pairwise_distances(self._features_batch1, self._features_batch2),
            )

            # Extra logic for less thans.
            self._radii1 = tf.placeholder(tf.float32, shape=[None, None])
            self._radii2 = tf.placeholder(tf.float32, shape=[None, None])
            dist32 = tf.cast(self.distance_block, tf.float32)[..., None]
            self._batch_1_in = tf.math.reduce_any(dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf.math.reduce_any(dist32 <= self._radii1[:, None], axis=0)

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.
        """
        return self.session.run(
            self.distance_block,
            feed_dict={self._features_batch1: U, self._features_batch2: V},
        )

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={
                self._features_batch1: batch_1,
                self._features_batch2: batch_2,
                self._radii1: radii_1,
                self._radii2: radii_2,
            },
        )


def _batch_pairwise_distances(U, V):
    """
    Compute pairwise distances between two batches of feature vectors.
    """
    with tf.variable_scope("pairwise_dist_block"):
        # Squared norms of each row in U and V.
        norm_u = tf.reduce_sum(tf.square(U), 1)
        norm_v = tf.reduce_sum(tf.square(V), 1)

        # norm_u as a column and norm_v as a row vectors.
        norm_u = tf.reshape(norm_u, [-1, 1])
        norm_v = tf.reshape(norm_v, [1, -1])

        # Pairwise squared Euclidean distances.
        D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            with zip_f.open(f"{zip_f.namelist()[0]}", "r") as arr_f:
                yield arr_f


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L50-L63
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


def polynomial_kernel(X, Y=None, degree=3, coef0=1, gamma=None):
    """
    Manually computes the polynomial kernel between two sets of features.
    
    :param X: First feature set (N x D).
    :param Y: Second feature set (M x D), if None, set Y = X (default).
    :param degree: Degree of the polynomial kernel.
    :param coef0: Independent term in the polynomial kernel.
    :param gamma: Kernel coefficient for polynomial kernel.
    :return: Kernel matrix of size N x M.
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]  # Default gamma value: 1 / number of features

    K = (gamma * np.dot(X, Y.T) + coef0) ** degree
    return K


def compute_kid(activations1: np.ndarray, activations2: np.ndarray, degree=3, coef0=1, gamma=None, eps=1e-6) -> float:
    """
    Compute Kernel Inception Distance (KID) using MMD between two sets of activations.
    
    :param activations1: Numpy array of features from the first set (e.g., real images).
    :param activations2: Numpy array of features from the second set (e.g., generated images).
    :param degree: Degree of the polynomial kernel (default: 3).
    :param coef0: Independent term in the polynomial kernel (default: 1).
    :param gamma: Kernel coefficient for RBF or polynomial kernel.
    :param eps: Small value to avoid division by zero.
    :return: The KID score (squared MMD).
    """
    
    # Compute kernel for both real and generated features
    k11 = polynomial_kernel(activations1, activations1, degree=degree, coef0=coef0, gamma=gamma)
    k22 = polynomial_kernel(activations2, activations2, degree=degree, coef0=coef0, gamma=gamma)
    k12 = polynomial_kernel(activations1, activations2, degree=degree, coef0=coef0, gamma=gamma)

    # Compute MMD between the two distributions
    m = activations1.shape[0]
    n = activations2.shape[0]

    # MMD estimate
    mmd = (np.sum(k11) / (m * (m - 1))) + (np.sum(k22) / (n * (n - 1))) - (2 * np.sum(k12) / (m * n))
    
    # Return squared MMD (KID estimate)
    return mmd
