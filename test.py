import os.path as osp
import glob
import cv2
import numpy as np
import torch
import discord
from discord.ext import commands
import RRDBNet_arch as arch

model_path = "C:/Users/danie/OneDrive/Dokumente/vislavision/ESRGAN/models/RRDB_PSNR_x4.pth"
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

bot = commands.Bot(command_prefix='/')

@bot.command()
async def scal(ctx, file_path):
    test_img_folder = file_path

    for path in glob.glob(test_img_folder):
        base = osp.splitext(osp.basename(path))[0]
        print(base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

        await ctx.send(file=discord.File('results/{:s}_rlt.png'.format(base)))

@client.command()
async def scal(ctx, *, file_path):
    # run image upscaling script
    model_path = "C:/Users/danie/OneDrive/Dokumente/vislavision/ESRGAN/models/RRDB_PSNR_x4.pth"
    device = torch.device('cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    
    # save the upscaled image
    filename = file_path.split("/")[-1]
    cv2.imwrite(f"upscaled_{filename}", output)
    
    # upload the upscaled image to discord channel
    await ctx.send(file=discord.File(f"upscaled_{filename}"))
import discord
from discord.ext import commands

TOKEN = 'MTEwMTMwNDk5MDUyNzY3MjM4MQ.GLGCZD.aFzNn5SPIqPuHQiqpmYPLYEiXPBYP6jcErYr9g'

bot = commands.Bot(command_prefix='/') # Command prefix hier ändern

@bot.event
async def on_ready():
    print(f'{bot.user} ist bereit!')

bot.run(MTEwMTMwNDk5MDUyNzY3MjM4MQ.GLGCZD.aFzNn5SPIqPuHQiqpmYPLYEiXPBYP6jcErYr9g)

import os
from waifu2x import waifu2x
from PIL import Image

@bot.command(name='scal')
async def scale_image(ctx, resolution: str, file_path: str):
    valid_resolutions = {'4k': (3840, 2160), 'fhd': (1920, 1080)}
    if resolution.lower() not in valid_resolutions:
        await ctx.send(f'Ungültige Auflösung. Verfügbare Auflösungen: {", ".join(valid_resolutions)}.')
        return

    if not os.path.exists(file_path):
        await ctx.send('Die Datei existiert nicht.')
        return

    _, file_extension = os.path.splitext(file_path)
    if file_extension not in ('.jpg', '.jpeg', '.png'):
        await ctx.send('Ungültiger Dateityp. Bitte verwende nur JPG-, JPEG- oder PNG-Dateien.')
        return

    try:
        upscale_resolution = valid_resolutions[resolution.lower()]
        w2x = waifu2x.Waifu2x(scale=2, model_dir='C:\Users\danie\OneDrive\Dokumente\vislavision\ESRGAN\models\RRDB_PSNR_x4.pth')  # Hier den Pfad zu deinem Modell angeben
        image = Image.open(file_path)
        upscaled_image = w2x.upscale(image, upscale_resolution)
        upscaled_image.save(f'upscaled{file_extension}')
        await ctx.send(file=discord.File(f'upscaled{file_extension}'))
    except Exception as e:
        await ctx.send(f'Ein Fehler ist aufgetreten: {e}')
