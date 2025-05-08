# Note
- The test split in the training dataset is used for evaluation during model training, rather than for final 
prediction on unknown examples (i.e., those without labels).
- If you manually specify the dataset splits, you should provide "train", "valid", and "test" sets. Otherwise, the data
will be automatically divided into train, valid, and test sets with a ratio of 8:1:1.
- The toy dataset contains only a few training examples for quick exploration by users. We recommend using more 
training examples when tackling real problems. For fine-tuning tasks, it is preferable to have hundreds to thousands or even hundreds of thousands of training examples. Generally speaking, the more training examples and the higher the quality of the training data, the better the model will perform and the more accurate the predictions will be.
- For the regression task, we recommend normalizing the labels so that they fall within the range of 0 to 1 (or -1 to 1).
