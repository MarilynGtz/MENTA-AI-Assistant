from __future__ import print_function

import datetime
import os.path
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.exceptions import GoogleAuthError
from googleapiclient.errors import HttpError

import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

import pywhatkit
import pyttsx3
import speech_recognition as sr
import threading
from playsound import playsound
import time
import webbrowser
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
import art


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
#empty list to add tasks
tasks = []

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
# Load a pre-trained model
model = load_model('chatbot_model.h5')

engine = pyttsx3.init()
engine_running = False # Start the speech synthesis engine in the background
listener = sr.Recognizer()
# Variable to track the last interaction time
last_interaction_time = time.time()
trigger_phrase = "hey Menta"  # Change this to your desired trigger phrase

# Function to display the robot GUI
def display_gui():
    robot_art = art.text2art("MENTA", font="block")
    print("--------------------------------------------------------------------------------------------------")
    print("|                                       MENTA GUI                                                |")
    print("--------------------------------------------------------------------------------------------------")
    print(robot_art)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
            
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_jason):
    tag = intents_list[0]['intent']
    list_of_intents = intents_jason['intents']
    #result = "I'm sorry, I didn't understand that."
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
    
def open_youtube_url(video_url):
    driver = webdriver.Chrome()
    driver.implicitly_wait(5)
    driver.get(video_url)


                
def process_command(command):
    global last_interaction_time
    global tasks
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    if "I'm sad" in command:
        print("I'm sorry to hear that you're feeling sad.")
        # Find the corresponding intent in the intents.json file
        for intent in intents['intents']:
            if intent['tag'] == 'sad':
                response = random.choice(intent['responses'])
                print("MENTA:", response)
                 # Call the open_youtube_url function directly
                video_url = "https://youtu.be/uE-1RPDqJAY?t=17"
                open_youtube_url(video_url)
                engine.say(response)
                engine.runAndWait()
                break
    
            process_command("I'm sad")   
        else:
            print("I'm here to help you. How can I assist you?")
        return
    
     # Check if the command is related to the calendar
    if "calendar" in command:
        try:
            # Call the Google Calendar API to retrieve upcoming events
            service = build('calendar', 'v3', credentials=creds)     
            now = datetime.datetime.utcnow().isoformat() + 'Z'
            events_result = service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=5,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            
            if not events:
                print('No upcoming events found.')
                engine.say('No upcoming events found.')
                engine.runAndWait()
            else:
                print('Upcoming events:')
                engine.say('Upcoming events:')
                for event in events:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    summary = event['summary']
                    print(start, summary)
                    engine.say(f'{start}: {summary}')
                    engine.runAndWait()
        
        except HttpError as error:
            print('An error occurred:', str(error))
            engine.say('An error occurred while fetching calendar events.')
            engine.runAndWait()

    elif "add to list" in command:
        # Extract the task description from the command
        task_description = command.replace("add to list", "").strip()
        print("Adding task to list:", task_description)
        engine.say(f"Adding task to list: {task_description}")
        engine.runAndWait()
        
        # Add the task to the list
        tasks.append(task_description)
    
    elif "what do i have on my list" in command:
        if len(tasks) == 0:
            print("You don't have any tasks.")
            engine.say("You don't have any tasks.")
            engine.runAndWait()
        else:
            print("Your tasks:")
            engine.say("Your tasks:")
            for task in tasks:
                print(task)
                engine.say(task)
                engine.runAndWait()        

    elif "delete task" in command:
        # Extract the task index from the command
        task_index = re.findall(r'\d+', command)
        if task_index:
            task_index = int(task_index[0]) - 1
            if task_index < len(tasks):
                deleted_task = tasks.pop(task_index)
                print("Deleted task:", deleted_task)
                engine.say(f"Deleted task: {deleted_task}")
                engine.runAndWait()
            else:
                print("Invalid task index.")
                engine.say("Invalid task index.")
                engine.runAndWait()
        else:
            print("Invalid command format. Please specify the task index.")
            engine.say("Invalid command format. Please specify the task index.")
            engine.runAndWait()

    elif "search" in command:
        # Extract the search query from the command
        query = command.replace("search", "").strip()
        print("Searching for:", query)
        engine.say(f"Searching for: {query}")
        engine.runAndWait()
        
        # Open Google and perform the search
        search_url = f"https://www.google.com/search?q={query}"
        webbrowser.open(search_url)

    elif "date" in command:
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        print("The current date is:", current_date)
        engine.say(f"The current date is: {current_date}")
        engine.runAndWait()
        
    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        print("The current time is:", current_time)
        engine.say(f"The current time is: {current_time}")
        engine.runAndWait()
        
    else:
        ints = predict_class(command)
        res = get_response(ints, intents)
        print("MENTA:", res)
        engine.say(res)
        engine.runAndWait()        
    
        
        
        return
                
    # Check if any intent has the "reminders" tag
    for intent in intents['intents']:
        if intent['tag'] == 'reminders':
            # Check if any user input matches the patterns for the reminders intent
            for pattern in intent['patterns']:
                if pattern in command:
                    # Trigger action for reminders
                    print("Don't forget your wallet, purse, and to lock the door! Have a great day Marilyn!")
                    engine.say("Don't forget your wallet, purse, and to lock the door! Have a great day Marilyn!")
                    engine.runAndWait()
                    # Perform your desired action here
                    return
    else:
        ints = predict_class(command)
        res = get_response(ints, intents)
        print("MENTA:", res)
        engine.say(res)
        engine.runAndWait()

        # Pause for a few seconds after speaking
        time.sleep(2)

def listen():
    global last_interaction_time
    display_gui()
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                voice = listener.listen(source)
                command = listener.recognize_google(voice).lower()
                print("You said:", command)
                process_command(command)

                if trigger_phrase in command:
                    last_interaction_time = time.time()  # Update the last interaction time

                    # Remove the trigger phrase from the command
                    command = command.replace(trigger_phrase, "").strip()    
                

                    if "ok bye" in command:
                        print("Goodbye!")
                        engine.stop() #stops engine synthesis
                        break

                    # Update the last interaction time
                    last_interaction_time = time.time()

                    display_gui()
                    while True:
                        command = input("User: ")
                        process_command(command)

 
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")
            time.sleep(2)
        except sr.RequestError:
            print("Sorry, my speech recognition service is currently unavailable.")
        finally:
            #after processing the command, call listen to continue listening
            #Check if the elapsed time exceeds 120 seconds
            if time.time() - last_interaction_time > 120:
                print("Inactive for 2 minutes. Exiting...")
                print("Goodbye!")
                engine.stop() # Stops engine synthesis
                break
listen()            


def start_listening():
    #while True:
        listen()

print("\u2764 MENTA")

# Start listening in a separate thread
listening_thread = threading.Thread(target=start_listening)
#listening_thread.daemon = True
listening_thread.start()


