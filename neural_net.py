import numpy as np

class FixedRBFLayer:
    def __init__(self, 
                 X_train, 
                 units, 
                 dist_type, 
                 seed):
        self.X_train = X_train
        self.units = units

        n = X_train.shape[0]
        if units > n:
            raise Exception(f"Expected hidden unit length < {n}. Actual hidden unit length: {self.units}")
        else:
            rng = np.random.default_rng(seed)
            self.centers = rng.choice(a=X_train, size=units, replace=False)
            if dist_type == "max":
                all_dist = []
                for x in X_train:
                    for c in self.centers:
                        all_dist.append(abs(x - c))
                max_dist = np.max(np.array(all_dist))
                self.sigma = max_dist / np.sqrt(2 * self.units)
            elif dist_type == "avg":
                all_dist = []
                for x in X_train:
                    for c in self.centers:
                        all_dist.append(abs(x - c))
                avg_dist = np.average(np.array(all_dist))
                self.sigma = 2 * avg_dist
            else:
                raise Exception(f"{dist_type} is not a valid distance type. Choose max or avg.")

layer = FixedRBFLayer(X_train=np.array([[1], [2], [3], [4], [5]]),
                      units=4,
                      dist_type="max",
                      seed=12345)