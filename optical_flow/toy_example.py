import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt




def get_volume_gradients(
    V: np.ndarray, xy_sigma: float, t_sigma: float, xy_radius: int, t_radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spatial and temporal gradients of the input volume V.
    V: input volume (t, x, y)
    Returns: Vx, Vy, Vt
    """
    Vx, Vy, Vt = [
        scipy.ndimage.gaussian_filter(
            V,
            sigma=(t_sigma, xy_sigma, xy_sigma),
            order=order,
            radius=(t_radius, xy_radius, xy_radius),
        )
        for order in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    ]
    return Vx, Vy, Vt

def get_flow_vector_lukas_kanade(
    Vx: np.ndarray, Vy: np.ndarray, Vt: np.ndarray, xy_radius: int, t_radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vx, Vy, Vt: volumes (t, x, y)
    xy_radius: spatial radius for uniform filter
    t_radius: temporal radius for uniform filter
    """

    Vx2 = Vx * Vx
    Vy2 = Vy * Vy
    VxVy = Vx * Vy
    VxVt = Vx * Vt
    VyVt = Vy * Vt
    Sxx, Syy, Sxy, Sxt, Syt = [
        scipy.ndimage.gaussian_filter(arr, sigma=(1,1,1), radius=(t_radius, xy_radius, xy_radius))
        for arr in [Vx2, Vy2, VxVy, VxVt, VyVt]
    ]

    det = Sxx * Syy - Sxy * Sxy
    u = -(Sxt * Syy - Syt * Sxy) / det
    v = -(Sxx * Syt - Sxy * Sxt) / det

    return u, v


# %%
# again, go through all frames and plot the flow vectors, with stride of 20


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
volume = load_images("./optical_flow/toy_problem")
print(volume.shape)



# %%
# post-process




# %%
Vx, Vy, Vt = get_volume_gradients(
    volume, xy_sigma=1, t_sigma=1, xy_radius=5, t_radius=2
)

u, v = get_flow_vector_lukas_kanade(Vx, Vy, Vt, xy_radius=5, t_radius=3)
#

# Do something!

# %%
### Apply optical flow - again,
# go through all frames and plot the flow vectors, with stride of 20

magnitutes = np.sqrt(u**2 + v**2)

# compute the minimum by taking 95th percentile of two near static frames, i.e. 29 and 30th
# min_thresh = np.percentile(magnitutes[28:30].flatten(), 95)
# print(min_thresh)

# estimate the min. background noise by computing the optical flow for the static parts of the image and taking the maximum magnigute as min threshold.

min_threshold = np.percentile(magnitutes[:, :50, :50], 99)
# volume[:, :50, :50]  = 0  # set the static part to zero to avoid plotting flow vectors there



from skimage.filters import threshold_otsu
# Apply Otsu thresholding per frame to create a mask for the optical flow vector magnitudes
masks = np.zeros_like(magnitutes, dtype=bool)
for t in range(magnitutes.shape[0]):
    # Flatten to 1D for thresholding, ignore NaNs or infs if present
    frame_mag = magnitutes[t]
    valid_mag = frame_mag[np.isfinite(frame_mag)]
    if valid_mag.size == 0:
        continue
    thresh = threshold_otsu(valid_mag)
    # pick 90th percentile as threshold to be more conservative
    # thresh = np.percentile(valid_mag, 90)
    masks[t] =  (frame_mag > min_threshold) & (frame_mag > thresh)
    


fig, ax = plt.subplots()
for t in range(0, volume.shape[0], 1):
    ax.clear()
    stride = 20
    ax.imshow(volume[t], cmap="gray")
    for y in range(stride, volume[t].shape[0] - stride + 1, stride):
        for x in range(stride, volume[t].shape[1] - stride + 1, stride):
            flow_vector = np.array([u[t, y, x], v[t, y, x]])
            # if np.linalg.norm(flow_vector) < 1:  # Only plot significant flow vectors
                # continue
            if not masks[t, y, x]:  # Skip vectors below the Otsu threshold
                continue
            ax.quiver(
                x,
                y,
                flow_vector[0],
                flow_vector[1],
                color="red",
                angles="xy",
                scale_units="xy",
                scale=1/10,
            )

    ax.set_title(f"Estimated Flow Vectors (Frame {t + 1})")
    ax.axis("off")
    plt.pause(0.1)