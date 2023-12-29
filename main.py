import argparse
import bz2
import pickle
import time
from collections import Counter
import re
import nltk
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from SentimentNet import *
from tools import *
#nltk.download('punkt')

parser = argparse.ArgumentParser(description='Train for review sentiment analyzing model')
parser.add_argument('-train', default=False, type=bool, help='mode')
parser.add_argument('-pred', default=False, type=bool, help='mode')
parser.add_argument('-model_name', required=True, help='Model name followed by device type')
parser.add_argument('-train_path', default="", help='training data path')
parser.add_argument('-test_path', default="", help='testing data path')
parser.add_argument('-pred_path', default="", help='prediction data path')
parser.add_argument('-dic_name', default="dictionary.pkl", help='dictionary name to be stored or used')
parser.add_argument('-num_train', default=100000, type=int, help='number of sentences used to train')
parser.add_argument('-num_test', default=10000, type=int, help='number of sentences used to test & validation in 50 to 50 rate')
parser.add_argument('-lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('-epochs', default=10, type=int, help='number of epochs')
parser.add_argument('-batch', default=1000, type=int, help='number of batch')
parser.add_argument('-seq_len', default=200, type=int, help='sequence length or input size')
opt = parser.parse_args()

seq_len = opt.seq_len  # The length that the sentences will be padded/shortened to
num_train = opt.num_train  # We're training on the first 800,000 reviews in the dataset
num_test = opt.num_test  # Using 200,000 reviews from test set
batch_size = opt.batch
learning_rate = opt.lr
epochs = opt.epochs
print_every = 1

output_size = 1
embedding_dim = 800
hidden_dim = 512
n_layers = 2
drop_prob = 0.1

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
device_type = "gpu" if is_cuda else "cpu"
model_path = opt.model_name + "_" + device_type + ".pt"

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def save_dictionary(dict):
    print("Saving dictionary...")
    # create a binary pickle file
    f = open(opt.dic_name, "wb")

    # write the python object (dict) to pickle file
    pickle.dump(dict, f)

    # close file
    f.close()


def load_dictionary():
    print("Loading dictionary...")
    # open a file, where you stored the pickled data
    file = open(opt.dic_name, 'rb')

    # dump information to that file
    dict = pickle.load(file)

    # close the file
    file.close()

    return dict


def load_parameters():
    # Open the file for reading
    print("Loading parameters:")
    with open("params.json", "r") as fp:
        # Load the dictionary from the file
        parameters = json.load(fp)

    # Print the contents of the dictionary
    print(parameters)

    return parameters


def get_device():
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU using")

    return device


def training(train_data_path, test_data_path):
    print("Loading data...")
    with open(train_data_path, "r", encoding="utf8") as file:
        train_file = file.readlines()
    with open(test_data_path, "r", encoding="utf8") as file:
        test_file = file.readlines()
    print("Finishing to load!")

    print("Starting preprocess...")
    train_file = [x for x in train_file[:num_train]]
    test_file = [x for x in test_file[:num_test]]

    # Extracting labels from sentences.
    train_labels = [0 if x.split(' ', 1)[0] == '__label__1' else 1 for x in train_file]
    train_sentences = [x.split(' ', 1)[1].strip() for x in train_file]

    test_labels = [0 if x.split(' ', 1)[0] == '__label__1' else 1 for x in test_file]
    test_sentences = [x.split(' ', 1)[1].strip() for x in test_file]

    # Some simple cleaning of data
    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d', '0', train_sentences[i])

    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d', '0', test_sentences[i])

    # Modify URLs to <url>
    for i in range(len(train_sentences)):
        if ('www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com'
                in train_sentences[i]):
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

    for i in range(len(test_sentences)):
        if ('www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in
                test_sentences[i]):
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

    words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences
    start = time.time()
    for i, sentence in progressBar(train_sentences, start, "Tokenizing", "complete", length=50):
        # The sentences will be stored as a list of words/tokens
        train_sentences[i] = []
        for word in nltk.word_tokenize(sentence):  # Tokenizing the words
            words.update([word])  # Converting all the words to lowercase
            train_sentences[i].append(word)


    # Removing the words that only appear once
    words = {k: v for k, v in words.items() if v > 1}
    # Sorting the words according to the number of appearances, with the most common word being first
    words = sorted(words, key=words.get, reverse=True)
    # Adding padding and unknown to our vocabulary so that they will be assigned an index
    words = ['_PAD', '_UNK'] + words
    # Dictionaries to store the word to index mappings and vice versa
    word2idx = {o: i for i, o in enumerate(words)}
    idx2word = {i: o for i, o in enumerate(words)}
    save_dictionary(word2idx)

    for i, sentence in enumerate(train_sentences):
        # Looking up the mapping dictionary and assigning the index to the respective words
        train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

    for i, sentence in enumerate(test_sentences):
        # For test sentences, we have to tokenize the sentences as well
        test_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in
                             nltk.word_tokenize(sentence)]

    train_sentences = pad_input(train_sentences, seq_len)
    test_sentences = pad_input(test_sentences, seq_len)

    # Converting our labels into numpy arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    split_frac = 0.5  # 50% validation, 50% test
    split_id = int(split_frac * len(test_sentences))
    val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
    val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

    train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
    val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
    test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    device = get_device()

    vocab_size = len(word2idx) + 1

    model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    counter = 0
    clip = 5
    valid_loss_min = np.Inf

    start_time = time.perf_counter()
    model.train()
    for i in range(epochs):
        h = model.init_hidden(batch_size, device)

        for inputs, labels in train_loader:
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size, device)
                val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                val_losses_mean = np.mean(val_losses)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                progress = counter * batch_size / (epochs * num_train) * 100
                print("Epoch: {}/{} |".format(i + 1, epochs),
                      "Progress: {:.2f}% |".format(progress),
                      "Loss: {:.6f} |".format(loss.item()),
                      "Val Loss: {:.6f} |".format(val_losses_mean),
                      "Rest Time: {}".format(calc_time_to_complete(elapsed_time, progress)), end="\r")

                if val_losses_mean <= valid_loss_min:
                    torch.save({
                        'step_info': {'epoch': i,
                                      'counter': counter,
                                      'progress': str(progress)+'%'},
                        'parameters': {'batch_size': batch_size,
                                       'num_train': num_train,
                                       'learning_rate': learning_rate,
                                       'epochs': epochs,
                                       'vocab_size': vocab_size,
                                       'output_size': output_size,
                                       'embedding_dim': embedding_dim,
                                       'hidden_dim': hidden_dim,
                                       'n_layers': n_layers,
                                       'drop_prob': drop_prob},
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_losses_mean,
                    }, model_path)
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'
                          .format(valid_loss_min, np.mean(val_losses)), end="\r")
                    valid_loss_min = val_losses_mean

    # Loading the best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size, device)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("")
    print("Test loss: {:.6f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))


def preprocess(data_path, batch_size):
    print("Loading data...")
    with open(data_path, "r", encoding="utf8") as file:
        test_file = file.readlines()

    print("Starting preprocess...")
    # Extracting labels from sentences.
    test_labels = [0 if x.split(' ', 1)[0] == '__label__1' else 1 for x in test_file]
    test_sentences = [x.split(' ', 1)[1].strip() for x in test_file]

    # Some simple cleaning of data
    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d', '0', test_sentences[i])

    # Modify URLs to <url>
    for i in range(len(test_sentences)):
        if ('www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in
                test_sentences[i]):
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

    word2idx = load_dictionary()
    for i, t_sentence in enumerate(test_sentences):
        # For test sentences, we have to tokenize the sentences as well
        test_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in nltk.word_tokenize(t_sentence)]

    test_sentences = pad_input(test_sentences, seq_len)

    # Converting our labels into numpy arrays
    test_labels = np.array(test_labels)
    # Converting into tensor-dataset
    test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return test_loader


def predict(data_path, model_path, batch_size=10):
    checkpoint = torch.load(model_path)
    parameters = checkpoint['parameters']
    vocab_size = parameters['vocab_size']
    output_size = parameters['output_size']
    embedding_dim = parameters['embedding_dim']
    hidden_dim = parameters['hidden_dim']
    n_layers = parameters['n_layers']
    drop_prob = parameters['drop_prob']

    data_loader = preprocess(data_path, batch_size)

    device = get_device()

    model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    h = model.init_hidden(batch_size, device)
    num_correct = 0
    counter = 0
    start_time = time.perf_counter()
    for inputs, labels in data_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

        counter += len(inputs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        progress = counter / len(data_loader.dataset) * 100
        current_acc = num_correct / len(data_loader.dataset) * 100
        print("Progress: {:.1f}% |".format(progress),
              "Current accuracy: {:.2f}% |".format(current_acc),
              "Rest Time: {}".format(calc_time_to_complete(elapsed_time, progress)), end="\r")

    conclusion_acc = num_correct / len(data_loader.dataset)
    print("")
    print("==============================================")
    print("Accuracy: {:.2f}%".format(conclusion_acc * 100))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if opt.pred:
        pred_path = opt.pred_path
        predict(pred_path, model_path, 20)
    elif opt.train:
        train_path = opt.train_path
        test_path = opt.test_path
        training(train_path, test_path)