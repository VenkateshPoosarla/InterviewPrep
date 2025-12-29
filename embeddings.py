import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations

import helper_utils

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)
np.random.seed(123)