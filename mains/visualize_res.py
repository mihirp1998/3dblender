import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from nbtschematic import SchematicFile
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def sigmoid(x):
	return 1/(1+np.exp(-x))

res = 64
thresh = 0.001
epoch = 7
plot_box = False

# log_path = 'results/CLEVR_' + str(res) + '_MULTI_LARGE_2/PNP-Net-dual_recon-reduced-16k-mae/samples/epoch-' + str(epoch) + '/'

log_path = 'results/CLEVR_' + str(res) + '_MULTI_LARGE_2/PNP-Net-dual_recon-reduced-16k-mae-object_weighted_loss/generated-samples/epoch-' + str(epoch) + '/'

log_path = 'results/CLEVR_' + str(res) + '_GEN/PNP-Net-voxel_input-dual_recon-reduced-10k-mae-only_tree/samples/epoch-' + str(epoch) + '/'

data = np.load(log_path + 'generated_voxels.npz')
blocks_in = data['inp']
blocks_out = data['out']
# blocks_pmean = data['pmeans']
# blocks_pvar = data['pvars']

with open(log_path + 'trees.pickle', 'rb') as f:
    trees = pickle.load(f)

# with open(log_path + 'bboxes.pickle', 'rb') as f:
#     bboxes = pickle.load(f)

for b_in, b_out, tree in zip(blocks_in, blocks_out, trees):
	b_out = b_out.squeeze()
	b_in = b_in.squeeze()
	print(b_out.max())
	print(b_out.min())
	# b_out = sigmoid(b_out)
	b_out[b_out > thresh] = 1
	b_out[b_out < thresh] = 0
	# box = (np.asarray(box) * 2).tolist()
	print(tree)
	# print(box)
	# print(b_in.shape)
	# print(b_out.shape)
	print('-'*10)


	if plot_box:
		vertices = []
		for b in box:
			z1 = [b[0],b[1],b[2]]
			z2 = [b[0],b[1],b[2]+b[5]]
			z3 = [b[0],b[1]+b[4],b[2]+b[5]]
			z4 = [b[0]+b[3],b[1]+b[4],b[2]+b[5]]
			z5 = [b[0]+b[3],b[1]+b[4],b[2]]
			z6 = [b[0]+b[3],b[1]+b[4],b[2]]
			z7 = [b[0],b[1]+b[4],b[2]]
			z8 = [b[0]+b[3],b[1]+b[4],b[2]]
			z9 = [b[0]+b[3],b[1],b[2]]
			z10 = [b[0]+b[3],b[1],b[2]+b[5]]
			z11 = [b[0]+b[3],b[1],b[2]]
			z12 = [b[0],b[1],b[2]]
			z13 = [b[0],b[1],b[2]+b[5]]
			z14 = [b[0]+b[3],b[1],b[2]+b[5]]
			z15 = [b[0]+b[3],b[1]+b[4],b[2]+b[5]]
			z16 = [b[0]+b[3],b[1],b[2]+b[5]]
			z17 = [b[0]+b[3],b[1],b[2]]
			z18 = [b[0],b[1],b[2]]
			z19 = [b[0],b[1]+b[4],b[2]]
			z20 = [b[0],b[1]+b[4],b[2]+b[5]]
			v = np.array([z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20])
			vertices.append(v)

	fig = plt.figure(figsize=(13,5))
	# fig.suptitle(tree)
	ax = fig.add_subplot(121, projection='3d')
	ax.set_title('Input')
	ax.voxels(b_in)
	ax = fig.add_subplot(122, projection='3d')
	ax.set_title('Output')
	ax.voxels(b_out)
	if plot_box:
		for v in vertices:
			ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
			ax.plot(v[:, 0], v[:, 1], v[:, 2], c='black')
	# plt.tight_layout()
	plt.show()
	plt.close()








