import numpy as np

class TargetGenerator:
    """
        Generate target heatmaps from given coords
    """
    def __init__(self, hm_size=[64,64], sigma=2) -> None:
        """
        Args:
            hm_size: [int, int]. Width and height of Target heatmap
            sigma: int. Variance of Target heatmap
        """
        self.hm_size = hm_size
        self.sigma = sigma

    def generate_target(self, coords):
        """
        Args:
            coords: numpy.array, shape: [N, 2]. Coordinates of keypoints
        Return:
            targets: numpy.array, shape: [N, hm_size[1], hm_size[0]]. Generated target heatmaps
            target_weights: numpy.array, shape: [N]. Weights of generated target heatmaps
        """
        num_joints = coords.shape[0]
        W,H = self.hm_size

        target_weights = (coords[:, 0]>=0) & (coords[:, 0]<W) & (coords[:, 1]>=0) & (coords[:, 0]<H)
        
        x = np.linspace(0, W-1, W)[np.newaxis, :]
        x = np.repeat(x, num_joints, axis=0)
        y = np.linspace(0, H-1, H)[np.newaxis, :]
        y = np.repeat(y, num_joints, axis=0)


        targets = np.exp(- ((x-coords[:, 0:1])[:, :, np.newaxis]**2+(y-coords[:, 1:])[:, np.newaxis, :]**2) / self.sigma**2)

        return targets, target_weights

if __name__ == "__main__":
    generator = TargetGenerator(hm_size=[8,8])
    test_coords = np.array([[-1, -1],
                            [4, 4],
                            ])
    test_targets, test_weights = generator.generate_target(test_coords)
    assert test_targets.shape == (2, 8, 8)
    assert test_targets[1, 4, 4] == 1
    assert (test_weights == np.array([False, True])).all()
