import os
import torch

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")


class EDA:

    def __init__(self, text_fpath: str, block_size=8, batch_size=4):

        self.block_size = block_size
        self.batch_size = batch_size

        with open(os.path.join(DATA_PATH, "text", text_fpath), "r") as file:
            raw_text = file.read()
            characters = sorted(list(set(raw_text)))
            self.map_stoi = {char: i for i, char in enumerate(characters)}
            self.map_itos = {i: char for i, char in enumerate(characters)}

        self.txt_data = torch.tensor(self.encode(raw_text), dtype=torch.long)

        train_size = int(self.txt_data.shape[0] * 0.9)
        self.train_data = self.txt_data[:train_size]
        self.test_data = self.txt_data[train_size:]

    def get_batch(self, split_type="train") -> tuple:

        """
        This function generates batches of data, from the test or train data, for training.
        Batches are stacked into tensors for parallel operations
        :param split_type: str, "train" or "test"
        :return: tuple, (x, y), <torch.Tensor, torch.Tensor>
        """

        data = self.train_data if split_type == "train" else self.test_data
        random_indices = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # Select random CONTIGUOUS chunks of characters from dataset.
        x = torch.stack([data[i:i + self.block_size] for i in random_indices])
        y = torch.stack([data[i+1:i + self.block_size+1] for i in random_indices])

        return x, y

    def encode(self, input_string) -> list:
        return [self.map_stoi[char] for char in input_string]

    def decode(self, input_int_list) -> str:
        return "".join([self.map_itos[i] for i in input_int_list])


    def run(self):

        x, y = self.get_batch(split_type="train")
        print(f"Input: {x}, {x.shape}")
        print(f"Output: {y}, {y.shape}")


def eda():

    """
    This function is a pure Exploratory-Data-Analysis task, where we load the text data and find out about
    its vocabulary and how to tokenize it
    :return:
    """

    # Load text data
    txt_data = None
    with open(os.path.join(DATA_PATH, "text", "tiny_shakespere.txt"), "r") as file:
        txt_data = file.read()

    # Create a list of unique characters and sort then.
    characters = sorted(list(set(txt_data)))
    print(f"Vocabulary size: {len(characters)}")
    print(f"Vocabulary {''.join(characters)}")

    map_stoi = {char: i for i, char in enumerate(characters)}
    map_itos = {i: char for i, char in enumerate(characters)}
    encode = lambda input_string: [map_stoi[char] for char in input_string]
    decode = lambda input_int_list: "".join([map_itos[i] for i in input_int_list])

    # Simple test in python
    message = "Hi there!"
    print(encode(message))
    print(decode(encode(message)))

    # Show how torch will see the entire text (as intergers!
    data = torch.tensor(encode(txt_data), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:1000])

    # Split data into test and train
    train_size = int(data.shape[0] * 0.9)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Block size is the size for each training chunk
    # - The transformer will never more than block_size inputs when predicting the next character
    block_size = 8
    small_train_data = train_data[:block_size+1]
    x = small_train_data[:block_size]
    y = small_train_data[1:block_size+1]  # block_size + 1 because you need the next character for the training of the last one
    for i in range(block_size):
        context = x[:i+1]
        target = y[i]
        print(f"When input is '{context}' the target is '{target}'")


    g = 0

def main():
    txt_fpath = os.path.join(DATA_PATH, "text", "tiny_shakespere.txt")
    eda_obj = EDA(text_fpath=txt_fpath)
    eda_obj.run()



if __name__ == "__main__":

    main()