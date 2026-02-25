import torch
import torch.nn as nn
import gradio as gr

modelData = torch.load('model.pth')

fMean = modelData['fm']
fSTD = modelData['fs']
parameters = modelData['parameters']

linear = nn.Linear(1,1)
linear.load_state_dict(parameters)

model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

def classifyTumor(size,fMean,fSTD,parameters):
       
    feature = torch.tensor([
        [size]
    ])
    X = (feature-fMean)/fSTD
    
    prob = model(X)

    if prob > .5:
        classify = 'Malignant'
    else:
        classify = 'Benign'
    print(prob,classify)
    return classify

with gr.Blocks() as iface:
    sizeInput = gr.Number(label = "Tumor Size:")
    output = gr.Text(label = "Classificationo:")
    sizeInput.change(fn = classifyTumor, inputs = [sizeInput], outputs = [output])
iface.launch()