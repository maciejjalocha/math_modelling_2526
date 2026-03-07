# %%
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

path = "./optical_flow/toy_problem"


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
a = load_images(path)
print(a.shape)

# Animate the images using matplotlib and pause
plt.figure()
for i in range(a.shape[0]):
    plt.imshow(a[i], cmap="gray")
    plt.title(f"Frame {i + 1}")
    plt.axis("off")
    plt.pause(0.1)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.random((50, 50, 50))

fig, ax = plt.subplots()

for i, img in enumerate(data):
    ax.clear()
    ax.imshow(img)
    ax.set_title(f"frame {i}")
    # Note that using time.sleep does *not* work here!
    plt.pause(0.1)


