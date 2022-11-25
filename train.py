import os
import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics
from dvclive.lightning import DVCLiveLogger

from params import *

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]
np.random.seed(seed)
np.random.shuffle(lines)
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0


# Set up the encoder.
# encoder_inputs = torch.nn.Linear(batch_size, num_encoder_tokens)
# encoder_lstm = torch.nn.LSTM(num_encoder_tokens, latent_dim)
# encoder = torch.nn.Sequential(encoder_inputs, encoder_lstm)
encoder = torch.nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)

# Set up the decoder.
# decoder_inputs = torch.nn.Linear(batch_size, num_decoder_tokens)
# decoder_lstm = torch.nn.LSTM(num_decoder_tokens, latent_dim)
# decoder = torch.nn.Sequential(decoder_inputs, decoder_lstm)
decoder = torch.nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
decoder_linear = torch.nn.Linear(latent_dim, num_decoder_tokens)
decoder_softmax = torch.nn.Softmax(dim=2)
dense = torch.nn.Sequential(decoder_linear, decoder_softmax)

# Define the model 
class LSTMSeqToSeq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dense = dense
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_decoder_tokens)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (x_encoder, x_decoder), y = batch
        encoder_outputs, (state_h, state_c) = self.encoder(x_encoder)
        # We discard `encoder_outputs` and only keep the states.
        decoder_outputs, (_, _) = self.decoder(x_decoder, (state_h, state_c))
        y_hat = self.dense(decoder_outputs)
        # Reshape y and y_hat
        y = y.flatten()
        y_hat = y_hat.flatten(end_dim=1)
        # Log metrics
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = self.acc(y_hat, y)
        self.log("step_train_loss", loss, prog_bar=True)
        self.log("step_train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_encoder, x_decoder), y = batch
        encoder_outputs, (state_h, state_c) = self.encoder(x_encoder)
        decoder_outputs, (_, _) = self.decoder(x_decoder, (state_h, state_c))
        y_hat = self.dense(decoder_outputs)
        y = y.flatten()
        y_hat = y_hat.flatten(end_dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = self.acc(y_hat, y)
        self.log("step_val_loss", loss, prog_bar=True)
        self.log("step_val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        return optimizer


# init the autoencoder
arch = LSTMSeqToSeq(encoder, decoder)

# load the data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoder_input_data, decoder_input_data,
            decoder_target_data):
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data

    def __len__(self):
        return len(self.encoder_input_data)

    def __getitem__(self, idx):
        target = self.decoder_target_data[idx].argmax(axis=1)
        return (self.encoder_input_data[idx], self.decoder_input_data[idx]), \
               target


combined_data = CustomDataset(encoder_input_data, decoder_input_data, decoder_target_data)
train_len = int(len(combined_data)*0.8)
val_len = len(combined_data) - train_len
train, val = torch.utils.data.random_split(combined_data, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)

live = DVCLiveLogger()
csv = pl.loggers.CSVLogger("logs")
trainer = pl.Trainer(max_epochs=epochs, logger=[csv])
trainer.fit(model=arch, train_dataloaders=train_loader,
        val_dataloaders=val_loader)

# weights_path = "model/weights.hdf5"
# os.makedirs("model", exist_ok=True)
# try:
#     model.load_weights(weights_path)
# except:
#     print(f"Unable to load weights from {weights_path}. Compiling new model.")
#     model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
#     optimizer = keras.optimizers.RMSprop(lr=lr)
#     model.compile(
#         optimizer="rmsprop",
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )


# metric = "val_accuracy"
# live = DVCLiveCallback(dir="results", report=None, resume=True)
# checkpoint = keras.callbacks.ModelCheckpoint(
#    weights_path,
#    save_best_only=True,
#    save_weights_only=True,
#    verbose=True,
#    monitor=metric)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=metric, 
#                                               patience=3,
#                                               verbose=True)
# early_stop = keras.callbacks.EarlyStopping(monitor=metric,
#                                            patience=10,
#                                            verbose=True)
# time_stop = tfa.callbacks.TimeStopping(seconds=300, verbose=1)
# hist = model.fit(
#     [encoder_input_data, decoder_input_data],
#     decoder_target_data,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_split=0.2,
#     callbacks=[live, checkpoint, reduce_lr, early_stop, time_stop]
# )
