import torch

data=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

correct=torch.tensor([3.6, 6.1, 8.4, 10.9, 13.8, 15.7, 18.2, 21.3, 23.9, 25.8])

w=torch.randn(1 ,requires_grad=True)
b=torch.randn(1,requires_grad=True)

def predict(x):
    prediction=(w*x)+b
    return prediction

def calculateLoss(x,y):
    loss=((predict(x)-y)**2).mean()
    return loss

def updateParameters(loss, learningrate, w, b):
    loss.backward()
    wgrad=w.grad
    bgrad=b.grad

    with torch.no_grad():
        w-= learningrate * wgrad
        b-= learningrate * bgrad
        b.grad.zero_()
        w.grad.zero_()

for i in range(1000):
    print(f'predicting: {predict(data)}')
    loss=calculateLoss(data, correct)
    print(f'loss: {loss}')
    updateParameters(loss, 0.001, w, b)
