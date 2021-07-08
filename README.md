# CBOW
Implementation of Word2Vec's Continuous Bag of Words model.

The Continuous Bag Of Words architecture takes the context of each word as input and tries to predict the word corresponding to the context. The order of context words does not influence prediction.

##  Model Training
All the word embeddings are trained using below parameters.

                         |      |
------------------------ | ---- |
Epochs                   |  30  |
Embedding dimension      |  300 |
Learning rate            |  0.5 |
Window size              |  10  |

