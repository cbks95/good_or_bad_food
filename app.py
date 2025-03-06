from fastai.vision.all import *
import gradio as gr

learn_inf = load_learner('export.pkl')

#categories = tuple(learn_inf.dls.vocab)
categories = ("Ungesundes 'Essen'", "Echtes Essen")

def classify_image(img):
    pred, idx, probs = learn_inf.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image()
label = gr.Label()
examples = ['good_food.jpg', 'bad_food.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(share=True)
