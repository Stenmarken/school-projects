import pickle 
import numpy as np
import matplotlib.pyplot as plt
from functions import ComputeCost, ComputeCostMultiple

def load_batch(path):
    """ Copied from the dataset website """
    with open(path, 'rb') as fo:
        dataset_dict = pickle.load(fo, encoding='bytes')
 
    return dataset_dict

def montage(W, parameters):
	""" Display the image for each label in W """
	llambda, n_epochs, num_batches, eta = parameters
	fig, ax = plt.subplots(2, 5)
	for i in range(2):
		for j in range(5):
			im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
			sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
			sim = sim.transpose(1, 0, 2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y=" + str(5 * i + j))
			ax[i][j].axis('off')
	#plt.set_text(f"lambda={llambda}, n_epochs={n_epochs}, num_batches={num_batches}, eta={eta}")
	parameters_obj = plt.text(0.5, 1, f"lambda={llambda}, n_epochs={n_epochs}, num_batches={num_batches}, eta={eta}", ha='center')
	parameters_obj.set_position([-70, 50])
	plt.show()

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def ComputeGradsNum(X, Y, P, W, b, lamda, h, use_compute_cost_multiple):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	if use_compute_cost_multiple:
		c, _ = ComputeCostMultiple(X, Y, W, b, lamda)
	else:
		c, _ = ComputeCost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		if use_compute_cost_multiple:
			c2, _ = ComputeCostMultiple(X, Y, W, b_try, lamda)
		else:
			c2, _ = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			if use_compute_cost_multiple:
				c2, _ = ComputeCostMultiple(X, Y, W_try, b, lamda)
			else:
				c2, _ = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1, _ = ComputeCostMultiple(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2, _ = ComputeCostMultiple(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1, _ = ComputeCostMultiple(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2, _ = ComputeCostMultiple(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def view_W_matrix(W, parameters):
	s_im = []
	for i in range(10):  # Python uses 0-based indexing
		im = np.reshape(W[i, :], (32, 32, 3), order='F')
		#im = (im - im.min()) / (im.max() - im.min())
		#im = np.transpose(im, (1, 0, 2))  # Permute the dimensions
		s_im.append(im)
	s_im = np.asanyarray(s_im)
	montage(s_im, parameters)

def displayimages(images):
	num_images = images.shape[3]

	num_cols = min(10, num_images)
	num_rows = (num_images // num_cols) + (1 if num_images % num_cols != 0 else 0)
	print("num_cols", num_cols)
	print("num_rows", num_rows)
	_ , axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
	print(axes)

	for i in range(num_images):
		ax = axes[i // num_cols, i % num_cols]
		ax.imshow(images[:, :, :, i])
		ax.axis('off')

	for i in range(num_images, num_rows * num_cols):
		ax = axes[i // num_cols, i % num_cols]
		ax.axis('off')
		ax.set_visible(False)

	plt.show()
