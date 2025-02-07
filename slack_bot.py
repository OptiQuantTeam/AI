import os
from dotenv import load_dotenv
import slack_sdk
from binance.client import Client
from binance.enums import *
import argparse

load_dotenv()
bot_token = os.getenv("bot_token")
client = slack_sdk.WebClient(token=bot_token)

user_id = "U0889HFLP9P"
slack_msg = f'<@{user_id}> hihi' 

response = client.chat_postMessage(
    channel="capstone-trading",
    text=slack_msg
)
print(response)