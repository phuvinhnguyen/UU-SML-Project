# Statistic Machine Learning - Project

## How to run scripts
### Command
```bash
python -m examples.<script>
```

### Example
```bash
python -m examples.train_XGBoost
```

## Implementation Instruction

### Dataset (./data/BikeDemandDataset)
- This dataset is used in the training and evaluation script
- The dataset is input to the **fit** and **eval** function of each model
- **fit** and **eval** function should handle the dataset before training or evaluating
- Each entry in the dataset is a tuple (features, label):
    - features: A tensor containing the input data (e.g., weather, time, and other features for bike demand prediction).
    - label: A tensor containing the target value (e.g., low_bike_demand encoded as an integer).

**Example how to use the dataset:**

```python
train_dataset = BikeDemandDataset('path_to_file.csv', data_type='train') # get the train split
val_dataset = BikeDemandDataset('path_to_file.csv', data_type='validation') # get the validation split

for i in train_dataset:
    print('feature: ', i[0]) # Output is a tensor array of size [15] -> an array of number
    print('label: ', i[1]) # Output is a numpy array of size [] -> just a number
    break
```

### Methods (./methods/)
- each file in this folder is a class of a ML method
- each method must include **fit** and **eval** functions
- input of **fit** and **eval** is the dataset in the previous part

### Examples (./examples/)
- each file in this folder is a script that performs training and evaluating

To run, consider the following example:

```bash
git clone https://github.com/phuvinhnguyen/UU-SML-Project.git
cd UU-SML-Project
python -m examples.train_XGBoost
```