import math
import numpy as np

def gaussian_filter(image, kernel_size=7, sigma=2):
    """
        image: [h, w, 3]
    """
    h, w, _ = image.shape

    # pad
    radius = kernel_size // 2
    padded_image = np.zeros(([h+2*radius, w+2*radius, 3]))
    padded_image[3:-3, 3:-3] = image

    # generate filter
    # filter = np.zeros([kernel_size, kernel_size])
    indexes = np.linspace(0, kernel_size-1, kernel_size)
    x = indexes[:, np.newaxis]
    y = indexes[np.newaxis, :]
    filter = np.exp(- ((x-radius)**2+(y-radius)**2) / sigma**2) / (math.sqrt(2 * math.pi) * sigma)
    filter = filter[:, :, np.newaxis]

    # perform filter
    for i in range(h):
        for j in range(w):
            image[i][j][:] = np.sum(np.sum(padded_image[i:i+2*radius+1, j:j+2*radius+1, :] * filter,
                                           axis=0),
                                    axis=0)

    return image

if __name__ == "__main__":
    test_image = np.ones([3, 3, 3])
    print(gaussian_filter(test_image)[:, :, 0])