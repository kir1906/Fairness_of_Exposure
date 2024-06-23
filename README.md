# Fairness-of-exposure-in-ranking-system

---
🔗 [Link to the research paper we have implemented](https://arxiv.org/abs/1802.07281)

🔗 [Link to the dataset](https://users.soe.ucsc.edu/~yiz/papers/data/YOWStudy/)

This research paper explores the crucial aspects of ranking systems, particularly focusing on the fairness of exposure. 

### Key Concepts:
- **Ranking and Exposure**: Discusses the definition of ranking as the process of ordering items (like search results or job applicants) based on relevance or other criteria, and exposure as the visibility these items receive in the ranking.
- **Fairness in Ranking**: Addresses the equitable treatment of individuals or groups within rankings, considering aspects such as bias avoidance, equal opportunity, and diversity.

### Formulating the Optimization Problem:
The presentation defines the primary goal as finding an optimal fair ranking that also maximizes utility, formulated as a linear optimization problem.

- **Utility Measures**: It details how utility is measured in rankings, discussing metrics such as Discounted Cumulative Gain (DCG), Precision, and Recall.

- **Probabilistic Ranking**: Due to the exponential number of possible rankings, the presentation introduces a probabilistic approach using a distribution over rankings to optimize expected utility.
  
- **Optimizing Fair Rankings**: It further elaborates on using linear programming for maximizing expected utility subject to various fairness constraints including Demographic Parity, Disparate Treatment, and Disparate Impact.

- **Sampling Rankings**: Explains the application of the Birkhoff-von Neumann decomposition to derive fair rankings from probabilistic distributions.

## Fairness Constraints in Ranking Systems
Fairness constraints are crucial for ensuring that automated ranking systems do not perpetuate or amplify biases. In our presentation, we delve into several types of fairness constraints that aim to balance exposure and utility in rankings. Below is an overview of each constraint discussed:

### 1. Demographic Parity
Demographic Parity is about ensuring that different groups receive similar levels of exposure regardless of their representation in the data. This constraint is useful in scenarios where the goal is to provide equal visibility to all groups, thereby mitigating bias that arises from disproportionate representation.

### 2. Disparate Treatment
Disparate Treatment addresses direct discrimination in ranking. It ensures that similar items from different groups receive similar treatment in the ranking process, irrespective of the group characteristics. This constraint is vital for scenarios where relevance and other merit-based criteria should not be overshadowed by group identity.

### 3. Disparate Impact
Disparate Impact focuses on the outcomes of the ranking process, aiming to equalize the impact across different groups. This is particularly important in contexts where the economic or social outcomes of ranking (like click-through rates or hiring decisions) significantly affect the opportunities available to individuals. The goal is to ensure that the impact of decisions derived from ranking systems is proportionate to the utility each group provides.


By integrating these fairness constraints into ranking systems, we aim to foster more equitable treatment and exposure across different groups, thereby supporting ethical AI practices.

--- 
