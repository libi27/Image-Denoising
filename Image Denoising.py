import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open(path, 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=20000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8), model_label="MVN noise-denoise restults"):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
            noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    plt.suptitle(model_label)
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.title("noise:"+str(noise_range[i]) + "\nMSE:"+ str(np.round(np.mean((crop_image(noisy_images[:, :, i],
                                                                                   patch_size) - cropped_original) ** 2),8)))
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
        plt.title("MSE:"+ str(np.round(np.mean((cropped_original - denoised_images[i]) ** 2),8)))



class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    # cov - a k-by-D-by-D tensor with the k different covariance matrices. the
    #     covariance matrices should be scaled versions of each other.

    cov -  a D-by-D matrix with the covariance matrix.
    mix - k-length probability vector for the mixture of the gaussians.
    r_y - k-length scale by which the covariance is multiplied
    """

    def __init__(self, cov, mix, r_y):
        self.cov = cov
        self.mix = mix
        self.r_y = r_y


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    means - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix, means):
        self.P = P
        self.vars = vars
        self.mix = mix
        self.means = means


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return np.sum(multivariate_normal.logpdf(X.T, mean=model.mean[:, 0], cov=model.cov, allow_singular=True))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    d, M = X.shape
    k = len(model.mix)
    ll = np.zeros([k, M])
    for i in range(k):
        curr_sigma = model.r_y[0, i] * model.cov
        log_p = multivariate_normal.logpdf(x=X.T,
                                           cov=curr_sigma,
                                           mean=np.zeros(d),
                                           allow_singular=True)
        log_p_pi = np.add(log_p, np.log(model.mix[0, [i]][0]))
        ll[[i], :] = log_p_pi

    return np.sum(logsumexp(ll.T, axis=1, keepdims=True).T)


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE
    # Don't understand what do I need to do in this function since for the ICA model I trained separately
    # d gaussian mixture models for which I use the log_likelihood during EM computation

def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    model = MVN_Model(mean=np.mean(X, 1, keepdims=True), cov=np.cov(X))
    ll = MVN_log_likelihood(X, model)
    print('MVN LogLikelihood: ', ll)
    return model


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """

    d, M = X.shape

    # Parameters Initialization
    mix_prob = np.random.rand(k)
    mix_prob = mix_prob / np.sum(mix_prob)

    log_r = np.log(np.matlib.repmat(mix_prob, 1, 1))
    log_pi = np.log(np.matlib.repmat(mix_prob, 1, 1))

    Sigma = np.cov(X)
    Sigma_inv = np.linalg.inv(Sigma)

    log_x_t_Sigma_x = np.log((X.T @ Sigma_inv @ X).diagonal())

    ll = []
    iteration = 0
    while True:
        # E-Step
        log_c_i = calculate_c_i(X, k, Sigma, log_pi, np.exp(log_r))

        # M-Step
        log_pi = (-np.log(M) + logsumexp(log_c_i, axis=1, keepdims=True)).T
        log_r = (-np.log(d) - logsumexp(log_c_i, axis=1, keepdims=True) +
                 logsumexp(log_c_i + np.tile(log_x_t_Sigma_x, (k, 1)), axis=1, keepdims=True)).T

        if np.isnan(log_c_i).any() or np.isnan(log_pi).any() or np.isnan(log_r).any():
            break

        # LogLikelihood calculation
        ll.append(GSM_log_likelihood(X, GSM_Model(cov=Sigma, mix=np.exp(log_pi), r_y=np.exp(log_r))))
        print("End of iteration number: {} with LogLikelihood {}".format(iteration, ll[iteration]))
        iteration += 1
        if len(ll) > 1 and np.abs(ll[-1] - ll[-2]) < 1e-2 or iteration > 200:
            break

    plt.figure()
    plt.title("GSM learning curve")
    plt.plot(np.asarray(ll).T, '.', MarkerSize=15)
    return GSM_Model(cov=Sigma, mix=np.exp(log_pi), r_y=np.exp(log_r))


def learn_ICA(X, k, graph_plot_iteration=3):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    d, M = X.shape
    # Parameters Initialization
    Sigma = np.cov(X)
    eigenvals, P = np.linalg.eigh(Sigma)
    s = P.T @ X
    mix = np.zeros([d, k])
    means = np.zeros([d, k])
    vars = np.zeros([d, k])
    plt.figure()
    plt.title("ICA learning curve")
    LL = np.zeros(d)
    for i in range(d):
        pi, mu, Sigma, ll = learn_GMM(s[[i], :], k=k)
        if i < graph_plot_iteration:
            plt.plot(np.asarray(ll).T, '.', MarkerSize=15, label='GMM learning in iter {}'.format(i))
            plt.legend()
        print("End of iteration number: {} with LogLikelihood {}".format(i, ll[-1]))
        mix[i, :] = pi
        means[i, :] = mu
        vars[i, :] = Sigma
        LL[i] = ll[-1]
    LL_sum = np.sum(LL)
    print("ICA LogLikelihood sum {}".format(np.sum(LL)))
    print("ICA LogLikelihood mean {}".format(LL_sum/d))
    return ICA_Model(P=P, vars=vars, mix=mix, means=means)


def learn_GMM(X, k):
    d, M = X.shape
    mix_prob = np.random.rand(k)
    mix_prob = mix_prob / np.sum(mix_prob)

    log_pi = np.log(np.matlib.repmat(mix_prob, 1, 1))

    Sigma = np.random.rand(1, k) * 0.00001
    sigma_factor = np.matlib.repmat(1, 1, 1)
    ll = []
    iteration = 0
    while True:

        # E-Step
        log_c_i = calculate_c_i(X, k, sigma_factor, log_pi, Sigma)
        c_i = np.exp(log_c_i)
        # M-Step
        ci_sum = np.sum(c_i, axis=1, keepdims=True).T
        pi = ci_sum / float(M)
        mu = (c_i @ X.T).T / ci_sum
        logsum_c_i = logsumexp(log_c_i, axis=1, keepdims=True)

        if np.isnan(log_c_i).any() or np.isnan(pi).any() or np.isnan(mu).any():
            break

        ll_iter = np.zeros([k, M])
        for y in range(k):
            x_centered = np.subtract(X, mu[0, y])
            x_mu = x_centered * x_centered
            log_Sigma = (logsumexp(log_c_i[[y], :] + np.log(x_mu), axis=1, keepdims=True) - logsum_c_i[[y], :]).T
            Sigma[0, y] = np.exp(log_Sigma)

            # LogLikelihood calculation
            log_p = multivariate_normal.logpdf(x=X.T,
                                               cov=Sigma[0, [y]],
                                               mean=mu[0, y],
                                               allow_singular=True)
            log_p_pi = np.add(log_p, np.log(pi)[0, [y]][0])
            ll_iter[[y], :] = log_p_pi

        ll.append(np.sum(logsumexp(np.asarray(ll_iter).T, axis=1, keepdims=True).T))
        iteration += 1
        if len(ll) > 1 and np.abs(ll[-1] - ll[-2]) < 1e-2 or iteration > 200:
            break

    return pi, mu, Sigma, ll


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    return weiner_filter(Y, mu=mvn_model.mean, Sigma=mvn_model.cov, noise_std=noise_std)


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """

    d, M = Y.shape
    k = len(gsm_model.mix)
    var = np.ones(d) * (noise_std ** 2)
    mu = np.zeros(Y[:, [0]].shape)
    Sigma = gsm_model.cov + var

    # Calculate C_i the posterior probability that our image patch came from Gaussian i.
    c_i = np.exp(calculate_c_i(Y, k, Sigma, np.log(gsm_model.mix), gsm_model.r_y))

    # Denoise the patches
    denoise_Y = np.zeros(Y.shape)
    for y in range(k):
        denoise_Y += c_i[:, y] * weiner_filter(Y, mu=mu, Sigma=gsm_model.r_y[0, y] * gsm_model.cov, noise_std=noise_std)
    return denoise_Y


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    s_noisy = ica_model.P.T @ Y
    denoise_Y = np.zeros(Y.shape)

    d, k = ica_model.means.shape

    for i in range(d):
        for j in range(k):
            denoise_Y[[i], :] += ica_model.mix[i, j] * weiner_filter(Y=s_noisy[[i], :],
                                                                     mu=ica_model.means[i, j].reshape([1, 1]),
                                                                     Sigma=ica_model.vars[i, j].reshape([1, 1]),
                                                                     noise_std=noise_std)

    return ica_model.P @ denoise_Y


def generate_training_set(path='train_images.pickle', patch_size=(8, 8), n=20000):
    """
    return a training set sampled by patches
    :param path: path to training image set
    :param patch_size:
    :param n:
    :return:
    """
    with open(path, 'rb') as f:
        train_pictures = pickle.load(f)

    return sample_patches(train_pictures, psize=patch_size, n=n)


def generate_test_set(path='test_images.pickle', remove_mean=True):
    """
    generate a test set
    :param path:
    :param remove_mean:
    :return:
    """
    with open(path, 'rb') as f:
        test_pictures = pickle.load(f)

    return grayscale_and_standardize(test_pictures, remove_mean)


def weiner_filter(Y, mu, Sigma, noise_std):
    """
    Wiener Filter for denoising problem as seen in TA
    x=(Sigma^-1 + (1/sigma_square)*I)^-1 @ (Sigma^-1@mu + (1/sigma_square)*y)
    :param Y:
    :param mu:
    :param Sigma:
    :param noise_std:
    :return:
    """
    Sigma_inv = np.linalg.inv(Sigma)
    sigma_noise = 1 / (noise_std ** 2)
    A = np.linalg.inv(Sigma_inv + sigma_noise * np.eye(Sigma_inv.shape[0]))
    B = (Sigma_inv @ mu) + (sigma_noise * Y)
    return A @ B


def calculate_c_i(X, k, Sigma, log_Pi, sigma_factor):
    """
    calculate the log of the posteriors used in EM algorithm
    :param X:
    :param k:
    :param Sigma:
    :param log_Pi:
    :param sigma_factor:
    :return:
    """
    d, M = X.shape
    log_C_i = np.zeros([k, M])
    for i in range(k):
        curr_sigma = sigma_factor[0, i] * Sigma
        log_p = multivariate_normal.logpdf(x=X.T,
                                           cov=curr_sigma,
                                           mean=np.zeros(d),
                                           allow_singular=True)
        log_C_i[[i], :] = np.add(log_p, log_Pi[0, [i]][0])
    log_C_i = log_C_i - logsumexp(log_C_i.T, axis=1, keepdims=True).T
    return normalize_log_likelihoods(log_C_i)


if __name__ == '__main__':
    # images_example()
    training_images = generate_training_set()
    test_images = generate_test_set()
    k_num = 2

    # MVN
    trained = learn_MVN(training_images)
    # for i in range(len(test_images)):
    #     test_denoising(test_images[:][i], trained, MVN_Denoise, model_label="MVN noise-denoise restults")
    test_denoising(test_images[:][0], trained, MVN_Denoise, model_label="MVN noise-denoise restults")


    # GSM
    trained = learn_GSM(training_images, k_num)
    # for i in range(len(test_images)):
    #     test_denoising(test_images[:][i], trained, GSM_Denoise, model_label="GSM noise-denoise restults")
    test_denoising(test_images[:][0], trained, GSM_Denoise, model_label="GSM noise-denoise restults")

    # ICA
    trained = learn_ICA(training_images, k_num)
    # for i in range(len(test_images)):
    #     test_denoising(test_images[:][i], trained, ICA_Denoise, model_label="ICA noise-denoise restults")
    test_denoising(test_images[:][0], trained, ICA_Denoise, model_label="ICA noise-denoise restults")
    plt.show()