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


# %%
from PIL.Image import new
import numpy as np
import av

container = av.open("./optical_flow/videos/apenture.mp4")
downscaled_shape = (640, 360)
i = 0
volume = []
for frame in container.decode(video=0):
    img = frame.to_image()  # PIL/Pillow image
    # arr = np.asarray(img)  # numpy array
    # aspect ratio is 16:9, so downscale to 160x90
    # downscale everything to 640x360
    # downscale everything to
    new_img = img.resize(downscaled_shape)
    arr = np.asarray(new_img)
    arr_gy = arr.mean(axis=2, dtype=np.float32) / 255.0  # cropp a little
    arr_gy = np.fliplr(arr_gy.T)  # ensure it is as recorderd
    volume.append(arr_gy)
volume = np.array(volume)


# %%
# post-process
volume = volume[:, :500, :]  # remove irrelevant right one
volume = volume[50:]  # drop first 50 frames as camera was sharpening




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
min_threshold = np.percentile(magnitutes[:, -50:, :50], 99)
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
    masks[t] = (frame_mag > thresh) & (frame_mag > min_threshold)
    


fig, ax = plt.subplots()
for t in range(0, volume.shape[0], 1):
    ax.clear()
    stride = 50
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