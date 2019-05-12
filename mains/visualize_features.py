

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from nbtschematic import SchematicFile
from mpl_toolkits.mplot3d import Axes3D

data = np.load('generated-samples/epoch-74/generated_voxels.npz')
inp = data['inp']
out = data['out']
del data

sample_idx = 12
thresh_in = 0.2
thresh_out = 0.2


for sample_idx in range(50):
	for channel_idx in range(32):
		b_in = inp[sample_idx, channel_idx]
		b_out = out[sample_idx, channel_idx]
		b_in = (b_in - b_in.min()) / (b_in.max() - b_in.min())
		b_out = (b_out - b_out.min()) / (b_out.max() - b_out.min())
		thresh_in = np.percentile(b_in, 95)
		thresh_out = np.percentile(b_out, 95)
		b_in[b_in < thresh_in] = 0.0
		b_out[b_out < thresh_out] = 0.0
		fig = plt.figure(figsize=(13,5))
		ax = fig.add_subplot(121, projection='3d')
		ax.set_title('Input')
		ax.voxels(b_in)
		ax = fig.add_subplot(122, projection='3d')
		ax.set_title('Output')
		ax.voxels(b_out)
		# plt.show()
		plt.savefig('voxels/sample_'+str(sample_idx)+'_channel_'+str(channel_idx)+'.jpg')
		plt.close()
		print(sample_idx, channel_idx)
