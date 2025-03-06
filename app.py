from fastai.vision.widgets import *
import gradio as gr

learn_inf = load_learner('export.pkl')

categories = tuple(learn_inf.dls.vocab)

def classify_image(img):
    pred, idx, probs = learn_inf.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['good_food.jpg', 'bad_food.jpg']

gr.Interface(fn=classify_image, inputs=image, outputs=label), examples=examples.launch(share=True)
