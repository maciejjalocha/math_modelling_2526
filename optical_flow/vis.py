# %%
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

path = "./optical_flow/toy_problem/"


# load all the images inside into 3d numpy array
def load_images(path):
    images = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".png"):
            img = plt.imread(os.path.join(path, filename))
            # convert to grayscale
            img = np.mean(img, axis=2, dtype=np.float32) / 255.0
            images.append(img)
    return np.array(images)


# Load images
frames = load_images(path)
print(frames.shape)


# %%
# Animate the images using matplotlib and pause
# plt.figure()
fig, ax = plt.subplots()
for i in range(frames.shape[0]):
    ax.imshow(frames[i], cmap="gray")
    ax.set_title(f"Frame {i + 1}")
    ax.axis("off")
    plt.pause(0.1)
# plt.show()

# %% [markdown]
# #### Problem 2.1 - Low level gradient calculation: The most simple way is to approximate the gradient
# as the difference between neighboring pixels/voxels.
# In a 1D signal, x, this would be:
# MATLAB: dx = x(2:end) - x(1:end-1)
# PYTHON: dx = x[1:] - x[0:-1]
# Obtain Vx,Vy,Vt for the entire volume and display for some selected frames.
# Consider what happens with the image/volume size, when performing gradient calculations like this?