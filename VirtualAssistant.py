import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import wikipedia
import datetime
import os
import time
import gtts
import speech_recognition as sr
from pynput.keyboard import Key, Listener
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import urllib.request
import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
import webbrowser
from tensorflow.python.framework import ops

with open("intents.json") as file:
    data = json.load(file)
try:
    x
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words :
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=4000, batch_size=8, show_metric=True)
model.save("model.tflearn")
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def IsItCommand(abc):
    if abc == 'youtube':
        abc = 1
        return abc
    elif abc == 'nmap':
        abc = 2
        return abc
    elif abc == 'alex':
        abc = 3
        return abc
    elif abc == 'wikipedia':
        abc = 4
        return abc
    else:
        return 0

def YoutubeCommand(inp):
    res = inp.partition("youtube")[2]
    print ("searcing on youtube:" + res)
    tts = gtts.gTTS("searching on youtube about " + res, lang="en")
    tts.save("youtube.mp3")
    os.system("mpg123 youtube.mp3")
    res.replace(" ", "+")
    driver = webdriver.Firefox()
    driver.get('https://www.youtube.com/results?search_query=' + res)
    searchmore = driver.find_elements_by_id("video-title")
    wait = WebDriverWait(driver, 10)
    wait.until(presence_of_element_located((By.ID, "video-title")))
    noumero = 0
    for i in searchmore:
        print(i)
        print(searchmore[noumero].text)
        noumero = noumero + 1

        wait.until(presence_of_element_located((By.ID, "iframe")))
        popup = driver.find_element_by_id("iframe")
        inppp = record_audio()
        print (inppp)
        numeric_filter = filter(str.isdigit, inppp)
        numeric_string = "".join(numeric_filter)
        clicked = int(numeric_string)
        print(clicked)
        searchmore[clicked].click()
        break

def Wiki_Search(inp):
    res = inp.partition("about")[2]
    print ("searcing on wiki:" + res)
    tts = gtts.gTTS("searching on wiki about" + res, lang="en")
    tts.save("hello.mp3")
    os.system("mpg123 hello.mp3")
    whatever = wikipedia.summary(res, sentences=5)
    tts = gtts.gTTS(whatever, lang="en")
    tts.save("hello.mp3")
    os.system("mpg123 hello.mp3")
    print(whatever)

r = sr.Recognizer()
def record_audio():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            voice_data = r.recognize_google(audio)
            print("Speech Recognition Results ---> " + voice_data)
        except sr.UnknownValueError:
            print("Did not get that")
            voice_data = "Unknown_Value"
        except sr.RequestError():
            print("Speech Recognition System Down")
            voice_data = "Recognizer_Down"
        return voice_data

def chat():
    while True:
        input("Press Enter to start listening...")
        inp = record_audio()
        print("!!!!!!!!" + inp)
        inp = inp.lower()
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print(tag)
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    flag = IsItCommand(tg['tag'])
                    print(flag)
                    if flag == 1:
                        YoutubeCommand(inp)
                        print("got it")
                        responses = tg['responses']
                    elif flag == 2:
                        os.system("sudo nmap -sS -O 192.168.0.1/24")
                        responses = tg['responses']
                    elif flag == 3:
                        driver = webdriver.Firefox()
                        driver.get('https://www.football-academies.gr/apo-ti-veroia-stin-italia-o-alexandros-iakovidis/')
                        responses = tg['responses']
                    elif flag == 4:
                        Wiki_Search(inp)
                        responses = tg['responses']
                    else:
                        responses = tg['responses']
            tts = gtts.gTTS(random.choice(responses), lang="en")
            tts.save("hello.mp3")
            os.system("mpg123 hello.mp3")
            print(random.choice(responses))
            print(responses)
        else:
            print ("I dont get that")
chat()
