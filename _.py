import matplotlib.pyplot as plt
import torch
import numpy as np

model_stuff = torch.load("test1.pt", map_location=torch.device('cpu'))
train_acc = model_stuff['train_acc']
val_acc = model_stuff['val_acc']
train_loss = model_stuff['train_list']
val_loss = model_stuff['val_list']

plt.plot(train_acc, label="Train Acc", color='red', marker='o')
plt.plot(val_acc, label='Val Acc', color='blue', marker='o')

plt.ylabel('Accuracy percentage of Loss')
plt.xlabel('Epochs')
plt.title('Image Contrastive Loss model')
plt.legend()

plt.show()
plt.close()

plt.plot(train_loss, label="Train loss", color='red', marker='o')
plt.plot(val_loss, label='Val loss', color='blue', marker='o')

plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Image Contrastive Loss model')
plt.legend()

plt.show()
plt.close()