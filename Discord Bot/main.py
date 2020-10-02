import discord
from Bot.bot import ChatBot

client = discord.Client()


@client.event
async def on_message(message):
    if not message.author.bot:
        reply = ChatBot.chat(message.content)
        await message.channel.send(reply)


def read_token():
    with open("token.txt") as f:
        lines = f.readlines()
        return lines[0].strip()


token = read_token()
print("The bot has started")
client.run(token)
