import sys
sys.path.append('..')
from sacred import Experiment

"""
Study the gap between filtered and non-filtered features.
The neural net has 16 hidden units.
Train/test splits vary from 1%-50%. 10% of train is used for early stopping.
Each split is run 10 times.
X1: Test accuracy of raw feature.
X2: Test accuracy of filtered feature (AugAdj^2). 
Y: Train ratio.
"""
ex = Experiment('filtering_gap')

@ex.automain
def run():
    clf_raw = Net('raw') 
    clf_filtered = Net('filtered')
    