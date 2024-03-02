import pandas as pd  # Importing pandas for data manipulation
import warnings

warnings.filterwarnings("ignore")


class NBMultinomial:
    def __init__(self):
        self.data = None  # Training data (X)
        self.target = None  # Target labels (y)
        self.probability_label = []  # List to store probabilities of labels

        self.classes_ = []  # Unique classes in the target data
        self.class_prior_ = None  # Prior probabilities of each class

    def train_data(self, X_data, y_data):
        # Function to train the model with input data and target labels
        self.target = y_data
        self.data = X_data
        self.process()  # Call process function to calculate class prior probabilities

    # Calculate the Likelihood
    def process(self):
        # Function to calculate prior probabilities of each class
        self.finding_classes()  # Find unique classes in the target data
        print(self.classes_)  # Print unique classes (optional)

        # Calculate prior probabilities of each class
        self.class_prior_ = [list(self.target).count(i) / len(self.target) for i in self.classes_]

    def finding_classes(self):
        # Function to find unique classes in the target data
        for i in self.target:
            if i not in self.classes_:
                self.classes_.append(i)
        self.classes_.sort()  # Sort the unique classes

    def predict(self, test_data):
        # Function to predict labels for test data
        predict_list = []  # List to store predicted labels
        for i in test_data.iloc:
            self.probability_label.clear()  # Clear probability list for each test sample
            for j in self.class_prior_:
                self.probability_label.append(j)  # Initialize probability list with class prior probabilities
            index_test_sample = i.name  # Get index of the test sample
            i = pd.DataFrame(i)  # Convert test sample to DataFrame
            i = i.where(i[index_test_sample] == 1).dropna()  # Filter non-zero values
            for j in i.index:
                self.calculate(j)  # Calculate probabilities for each word in the test sample

            # Predict label based on maximum probability
            label = self.classes_[self.probability_label.index(max(self.probability_label))]
            predict_list.append(label)  # Append predicted label to the list

        return predict_list  # Return list of predicted labels

    def calculate(self, word_number):
        # Function to calculate probabilities for each class based on a given word
        number_for_each_label = {key: 0 for key in self.classes_}  # Initialize count for each class

        # Count occurrences of the word in each class
        for i in self.data.iloc:
            if i[word_number] == 1:
                number_for_each_label[self.target[i.name]] += 1

        # Laplace smoothing: Add one to each count where the count is zero
        for i in number_for_each_label.values():
            if i == 0:
                number_for_each_label = {key: number_for_each_label[key] + 1 for key in number_for_each_label}

        # Calculate probabilities for each class
        self.probability_label = [self.probability_label[i] *
                                  (list(number_for_each_label.values())[i]
                                   / sum(number_for_each_label.values())) for i in range(len(number_for_each_label))]

    @staticmethod
    def score(predict_list, true_list):
        # Function to calculate accuracy score
        zip_list = zip(predict_list, true_list)
        true_number, false_number = 0, 0

        # Count number of correct and incorrect predictions
        for i in zip_list:
            if i[0] == i[1]:
                true_number += 1
            else:
                false_number += 1
        return true_number / len(true_list)  # Return accuracy score
