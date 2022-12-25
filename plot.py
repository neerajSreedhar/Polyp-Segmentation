import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv_file = r'files\\data.csv'

df = pd.read_csv(csv_file)
fig = plt.figure(figsize=(18, 6))
fig = plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(df['acc'], label='Train Acc')
plt.plot(df['val_acc'], label='Val Acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(df['loss'], label='Train loss')
plt.plot(df['val_loss'], label='Val loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.savefig('history_curves.png')