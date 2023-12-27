import time

import translators as ts
from tqdm import tqdm
import argparse
import os
import socket


def check_internet_connection():
    remote_server = "www.bing.com"
    port = 80
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect((remote_server, port))
        return True
    except socket.error:
        return False
    finally:
        sock.close()


parser = argparse.ArgumentParser(description='Train for review sentiment analyzing model')
parser.add_argument('-data_path', default="data/train/train.ft.txt", help='path for training data file')
opt = parser.parse_args()


writting_intervall = 100

if __name__ == '__main__':
    data_path = opt.data_path
    path = "/".join(data_path.split("/")[:-1]) + "/"
    file_name = data_path.split("/")[-1]

    path = path.replace("//", "/")

    translated_data_path = path + "de_" + file_name

    with open(data_path, "r", encoding="utf8") as file:
        sentences = file.readlines()

    if len(sentences) < writting_intervall:
        writting_intervall = len(sentences)
        print("writting_intervall is adjusted to {}".format(str(len(sentences))))

    already_translated_sentences = []
    if os.path.isfile(translated_data_path):
        with open(translated_data_path, "r", encoding="utf8") as file:
            already_translated_sentences = file.readlines()
        print("Existing already translated data: {} with {} sentences".format(translated_data_path, len(already_translated_sentences)))
    else:
        file = open(translated_data_path, "w", encoding="utf8")
        file.write("")
        file.close()

    failed_sentences = []
    translated_sentences = []
    temp_sentences = []
    count = 0
    for sent in tqdm(sentences, desc="Translating "+data_path):
        temp_sent = sent.split(" ", 1)
        label = temp_sent[0]
        text = temp_sent[1]
        try:
            while check_internet_connection() is not True:
                print("[Attention] Internet is not available now, waiting for connection...", end="\r")
                time.sleep(60*5)

            while len(text) > 1000:
                text = text.split(".")[:-1]
                text = ".".join(text)
                if len(text) == 0:
                    text = temp_sent[1][:1000]

            translated = ""
            for i in range(3):
                try:
                    translated = ts.translate_text(query_text=text, from_language="en", to_language="de")
                    break
                except Exception as err:
                    print("{}/{} [Error] Translation failed: {}".format(i, 3, err.message))
                    time.sleep(5)
            if len(translated) == 0:
                print(sent)
                continue

            labeled_translated_text = label + " " + translated.strip()
            if len(already_translated_sentences) > 0:
                if labeled_translated_text in already_translated_sentences:
                    continue
            translated_sentences.append(labeled_translated_text)
            count += 1
            if count >= writting_intervall:
                with open(translated_data_path, "a", encoding="utf8") as file:
                    for t_sent in translated_sentences:
                        file.write(t_sent + "\n")
                translated_sentences.clear()
                count = 0
        except:
            failed_sentences.append(sent)

    if len(translated_sentences) > 0:
        with open(translated_data_path, "a", encoding="utf8") as file:
            for t_sent in translated_sentences:
                file.write(t_sent + "\n")

    print("Translating is done")
    print("==========================================")
    print("Translation stored: {}".format(translated_data_path))
    print("Translating failed sentences: {}".format(len(failed_sentences)))
    if len(failed_sentences) > 0:
        with open(path + "failed_" + file_name, "w", encoding="utf8") as file:
            for sent in failed_sentences:
                file.write(sent)
