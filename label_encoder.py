import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings


class label_encoder:
    def __init__(self):
        self.unique_words = {}  # Dictionary to store unique words and their corresponding codes
        self.word_coder = 0  # Counter for assigning codes to words

    def train(self, train_data):
        # Function to train the label encoder on the training data
        for i in train_data.iloc:
            i = self.edit_sample(i)  # Preprocess the sample
            for j in i:
                if j not in self.unique_words.values():
                    self.unique_words[self.word_coder] = j  # Assign code to new unique word
                    self.word_coder += 1  # Increment word code

    def transform(self, transform_data):
        # Function to transform the input data using the trained label encoder
        indexes, all_data = [], []
        for i in transform_data.iloc:
            indexes.append(i.name)  # Store the index of each sample
            i = self.edit_sample(i)  # Preprocess the sample
            sample = []
            # Iterate over each word in the sample and encode it if it exists in the unique words dictionary
            [sample.append(list(self.unique_words.values()).index(j)) for j in i if j in self.unique_words.values()]
            all_data.append(sample)  # Append the encoded sample to the list

        return self.create_dataframe(all_data, indexes)  # Create a DataFrame from the encoded data

    def create_dataframe(self, transformed_data, index_set):
        # Function to create a DataFrame from the transformed data
        transformed_data = pd.DataFrame(transformed_data, index=index_set)  # Create DataFrame from encoded data

        # Create a DataFrame with columns for each unique word and fill with zeros
        final_data_frame = pd.DataFrame(columns=[i for i in range(len(self.unique_words))], index=index_set)
        final_data_frame = final_data_frame.fillna(0)

        # Fill the final DataFrame with 1s where words are present in the encoded data
        for i in transformed_data.iloc:
            i = i.loc[~i[:].isna()]  # Remove NaN values
            for j in i:
                final_data_frame.at[i.name, j] = 1  # Set corresponding entry to 1

        return final_data_frame  # Return the final DataFrame

    @staticmethod
    def edit_sample(sample):
        # Function to preprocess the input sample
        for j in sample:
            # Replace punctuation marks with spaces and convert to lowercase
            j = str(j).replace("...", " ").replace("!", " ").replace("?", " ").replace(",", " ").replace(". ", " ").lower()
            return str(j).split(" ")  # Split the sample into words and return as a list

