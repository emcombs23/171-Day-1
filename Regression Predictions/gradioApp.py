import gradio as gr
import torch
import torch.nn as nn

def calculateOutput(size, featureMean, featureSTD, targetMean, targetSTD, parameters):
    features = torch.tensor([
        [size]
    ])

    X = (features-featureMean)/featureSTD

    model = nn.Linear(1,1)
    model.load_state_dict(parameters)

    prediction = model(X)

    result = prediction*targetSTD+targetMean
    return result.item()

modelData = torch.load('model.pth')

featureMean = modelData['fm']
featureSTD = modelData['fs']
targetMean = modelData['tm']
targetSTD = modelData['ts']

parameters = modelData['parameters']

with gr.Blocks() as iface:
    sizeInput = gr.Number(label = "Square Footage Input:")
    output = gr.Number(label = "Price Output:")

    sizeInput.change(fn = calculateOutput, inputs = [sizeInput, featureMean, featureSTD, targetMean, targetSTD, parameters], outputs = [output])
iface.launch()