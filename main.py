import sys
import os
import discord
import logging

# mute warnings
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

print("Trying to import tf\n")
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow import keras

print("Imported tf")
import numpy as np


client = discord.Client()

with open("log.txt", "w") as file:
    file.write("\n")
model = keras.models.load_model("model.h5")

# old version of numpy must be used
# pip install numpy==1.16.1
imdb = keras.datasets.imdb

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=88000)


# print(train_data[0])
# data is in integer encoded words

# A dictionary mapping words to an integer index
_word_index = imdb.get_word_index()


# adding3 to each int value for custom vals
word_index = {k: (v + 3) for k, v in _word_index.items()}

# values of that extra space
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# swapping keys and values in the wordindex
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# function taht actually decodes it
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def review_encode(s):
    encoded = [1]
    # encodes evry word by looping and assigning it a number value.
    # then addes it to the encdoe

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            # adds the unkown tag if it is unkown
            encoded.append(2)

    return encoded


print("\n\n\n\n\n")


@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    channel = message.channel

    message_text = str(message.content).lower()

    other_than_trigger_word_list = (message_text.split())[1:]

    print("Pure text: " + str(message_text))

    if message_text.startswith("$predict"):

        full = "".join(other_than_trigger_word_list)

        with open("log.txt", "a") as file:
            print("Writing the prediction " + str(full) + " to log.txt")
            file.write(str(full) + "\n")

        # make a prediction for each reviw(1 reveiw is one line[file is i reveiw and 1 line])

        # removin extra chars to spliy into words
        nline = (
            full.replace(",", "")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace('"', "")
            .replace("\n", "")
            .strip()
            .split(" ")
        )
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences(
            [encode], value=word_index["<PAD>"], padding="post", maxlen=250
        )  # make the data 250 words long

        predict = model.predict(encode)

        # nline is an array of words
        print(nline)
        # print(encode)
        percent = float(predict[0][0])
        print("Percentage out of 1: " + str(percent))

        # print a separator

        # multiplied it by 100 to actually make it a percent(bef it was out of 1 now it is ot of 100)
        percenttosubmit = str(round(percent * 100, 2)) + "%"

        type_percent = ""
        if float(percent) > 0.5:
            type_percent = "positive"
        else:
            type_percent = "negative"

        await channel.send(type_percent)
        await channel.send(percenttosubmit)
    elif message_text.startswith("$help"):
        print("Sening help message:")
        help_text = "This is a prediction bot. Use $predict at the beggining of your message, and everything after will be sent  my prediction mode. It will return its prediction on if the text is postive (happy, good etc) or if it is negative (sad, bad, etc"

        await channel.send(help_text)

    # print a separator
    print("-" * 50)


# @client.event
client.run("NjQ3NDg4MjYxMjAzODUzMzUy.XoIobw.VcKms_eYSZskp0Rswr7CoTrKlRo")
