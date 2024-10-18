import pickle 
import numpy as np
import matplotlib.pyplot as plt

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
