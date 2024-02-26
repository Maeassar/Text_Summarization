import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BART_df = pd.read_csv("Seq2SeqBART_loss.csv")
BART_loss = BART_df["Loss"]

LSTM_df = pd.read_csv("Seq2SeqLSTM_loss.csv")
LSTM_loss = LSTM_df["Loss"]

plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(BART_loss, label="BART", color="blue")

ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.set_title("BART Loss")


ax2.plot(LSTM_loss, label="LSTM", color="green")

ax2.set_xlabel("Steps")
ax2.set_ylabel("Loss")
ax2.set_title("LSTM Loss")

plt.show()