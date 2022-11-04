from fastai.vision.all import *
import gradio as gr
import requests
import base64
from bs4 import BeautifulSoup
import os

# Load the trained model
learn = load_learner('nsfw_model.pkl')
labels = learn.dls.vocab

def analyze(url):
    """Analyzer function that classifies the images found at the given URL"""
    
    # Make sure URL starts with http or https
    # TODO: confirm that the url points to a web page, and not some resource.
    # Regex could be useful here
    if not url.startswith(('http://','https://')):
        url = 'http://'+url
    
    safety = 'safe' # our return variable

    # Extract html and all img tags
    html = requests.get(url)
    soup = BeautifulSoup(html.text, "html.parser")
    img_elements = soup.find_all("img")

    # Save all src urls that we can clearly tell are img urls.
    # A better approach would be to use regex here
    srcs = []
    for img in img_elements:
        for v in img.attrs.values():
            if isinstance(v, str):
                if v.lower().endswith(('jpg', 'png', 'gif', 'jpeg')):
                    srcs.append(v)
    
    # Get the images from the urls and classify
    # If there is a single unsafe image, report it.
    for src_url in srcs:
        try:
            img_data = requests.get(src_url).content
            temp = 'temp.' + src_url.lower().split('.')[-1]
            with open(temp, 'wb') as handler:
                handler.write(img_data)
            is_nsfw,_,probs = learn.predict(PILImage.create(temp))
            os.remove(temp) 
            if is_nsfw == "unsafe_searches":
                safety = 'NOT safe'
                return safety
        except Exception as e:
            pass
    return safety

title = "Website Safety Analyzer"
description = "**The internet is not safe for children**. Even if we know the 'bad' sites, social media is hard to regulate.  \n"+\
                "This is step one in an attempt to solve that. An image classifier that audits every image at a URL.  \n"+\
                "In this iteration, I classify sites with sexually explicit content as **'NOT safe'**.  \n\n"+\
                "There is a long way to go with NLP for profanity, cyber-bullying, as well as CV for violence, substance abuse, etc.  \n"+\
                "I welcome any help on this. 🙂"
examples = ['porhub.com', 'cnn.com', 'xvideos.com', 'www.pinterest.com']
enable_queue=True

iface = gr.Interface(
    fn=analyze, 
    inputs="text", 
    outputs="text",
    title=title,
    description=description,
    examples=examples,
)
iface.launch(enable_queue=enable_queue)