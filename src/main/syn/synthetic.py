import pandas as pd
from func import show_rankings_1
from func import show_rankings_2


"""#Synthetic Data Experiment

## Method 1 (Using SCS(CVXPY) solver())
"""

sol = pd.DataFrame({
            'U': [0.81, 0.6, 0.7,0.4,0.5,0.3],
            'G': [0,0,0,1,1,1],
            })
exp_df = show_rankings_1(sol['U'],sol['G'],3,5)

"""# Method 2 (Not So Optimal)"""


exp_df = show_rankings_2(sol['U'],sol['G'])
