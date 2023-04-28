import discord
from discord.ext import commands

TOKEN = 'MTEwMTMwNDk5MDUyNzY3MjM4MQ.GLGCZD.aFzNn5SPIqPuHQiqpmYPLYEiXPBYP6jcErYr9g'

bot = commands.Bot(command_prefix='/') # Command prefix hier ändern

@bot.event
async def on_ready():
    print(f'{bot.user} ist bereit!')

bot.run(TOKEN)