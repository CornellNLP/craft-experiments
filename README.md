# CRAFT experiments

Training the [CRAFT model](https://aclanthology.org/D19-1481.pdf) to enable generative pretraining. To this end, encode
the conversation as a single document and feed that straight to a transformed model as a standard text classification
task (predict next token). The argument is that the hierarchical structure is really not necessary.

### References:

* Using BERT instead of CRAFT: [Dynamic Forecasting of Conversation Derailment](https://arxiv.org/abs/2110.05111),
  includes a direct comparison to CRAFT (verdict: the BERT model gets performance gains, but perhaps more modest than
  you might expect) and additional inferences on static vs. dynamic training.
* Using regression head to predict the derailment distance, in addition to the classification head (includes
  transformers
  and hierarchical model): [Conversation Modeling to Predict Derailment](https://arxiv.org/pdf/2303.11184.pdf).

---

Author: Tushaar Gangavarapu (tg352)
