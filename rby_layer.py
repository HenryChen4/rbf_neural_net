import numpy as np

class FixedRBFLayer:
    def __init__(self,  
                 units, 
                 dist_type, 
                 seed):
        self.units = units

    # input shape: (n, 1)
    # x = [[],
    #      [],
    #      []]
    def initialize(self, X_train):
        self.X_train = X_train

        # choose centers and sigmas randomly
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

    def feed_forward(self, x_in):
        h = np.array([])
        for c_j in self.centers:
            d_j = 0
            for x_i in x_in:
                d_j += ((x_i - c_j) ** 2)
            d_j = np.sqrt(d_j)            
            h_j = np.exp(-(d_j**2)/(2 * (self.sigma ** 2)))
            h.append(h_j)
        return h_j