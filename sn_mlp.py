import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
# Dati
df = pd.read_csv('new_rich.csv')
energia = df['energy'].values
flusso = df['flux'].values
FWHM = df['FWHM'].values
snr = df['snr'].values
time = df['time'].values
di = df['distance'].values
dati = np.stack([energia, flusso, FWHM, snr, di], axis=-1)  # Forma: (num_dati, 5)
etichette = df['model'].values

# Standardizzazione dei dati
scaler = StandardScaler()
dati_senza_distanze = dati[:, :-1]
dati_senza_distanze_scaled = scaler.fit_transform(dati_senza_distanze)
dati_scaled = np.concatenate((dati_senza_distanze_scaled, dati[:, -1:]), axis=1)

# Divisione in training e test set
dati_train_scaled, dati_test_scaled, etichette_train, etichette_test, distanze_originali_train, distanze_originali_test = train_test_split(
    dati_scaled, etichette, di, test_size=0.2, random_state=42)

# Conversione a tensori PyTorch
dati_train = torch.from_numpy(dati_train_scaled).float()
dati_test = torch.from_numpy(dati_test_scaled).float()
etichette_train = torch.from_numpy(etichette_train).long()
etichette_test = torch.from_numpy(etichette_test).long()
di_test = torch.from_numpy(distanze_originali_test).float()

# Creazione di DataLoader per batching
train_dataset = TensorDataset(dati_train, etichette_train)
test_dataset = TensorDataset(dati_test, etichette_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Definizione del modello MLP
class ModelloMLP(nn.Module):
    def __init__(self):
        super(ModelloMLP, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # Input: 5 features, Output: 64 neuroni
        self.fc2 = nn.Linear(64, 128) # Input: 64 neuroni, Output: 128 neuroni
        self.fc3 = nn.Linear(128, 5) # Input: 128 neuroni, Output: 5 classi
        self.dropout = nn.Dropout(0.5) #Dropout per regolarizzazione

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

modello = ModelloMLP()

# Definizione della loss e dell'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modello.parameters())

# Addestramento del modello
num_epochs = 70 #Aumentato il numero di epoche per una migliore convergenza
for epoch in range(num_epochs):
    for dati_batch, etichette_batch in train_loader:
        optimizer.zero_grad()
        outputs = modello(dati_batch)
        loss = criterion(outputs, etichette_batch)
        loss.backward()
        optimizer.step()

# Valutazione del modello
with torch.no_grad():
    correct = 0
    total = 0
    confidenze = []
    distanze_test_lista = [] 
    for i, (dati_batch_test, etichette_batch_test) in enumerate(test_loader):
        outputs = modello(dati_batch_test)
        _, predicted = torch.max(outputs.data, 1)
        total += etichette_batch_test.size(0)
        correct += (predicted == etichette_batch_test).sum().item()
        #Calcolo delle confidenze
        softmax = nn.Softmax(dim=1)
        probabilita = softmax(outputs)
        max_prob, _ = torch.max(probabilita, dim=1)
        confidenze.extend(max_prob.tolist())

        #Esempio di come aggiungere le distanze (da adattare al tuo caso)
        indice_inizio = i * test_loader.batch_size
        indice_fine = min((i+1)*test_loader.batch_size, len(di_test))
        distanze_batch = di_test[indice_inizio : indice_fine].numpy()
        distanze_test_lista.extend(distanze_batch.tolist())
#print(distanze_test_lista)
print('Accuracy del modello sui dati di test: {} %'.format(100 * correct / total))

#Analisi della confidenza in funzione della distanza
distanze_test = np.array(distanze_test_lista)
confidenze = np.array(confidenze)
soglia_confidenza = 0.7  # Soglia di confidenza

# Ordina le distanze e le confidenze in base alle distanze
indici_ordinati = np.argsort(distanze_test)
distanze_ordinate = distanze_test[indici_ordinati]
confidenze_ordinate = confidenze[indici_ordinati]
# Trova la distanza limite
indici_sotto_soglia = np.where(confidenze_ordinate < soglia_confidenza)[0]

if len(indici_sotto_soglia) > 0: #Verifico se ci sono distanze con confidenza sotto la soglia
    distanza_limite = distanze_ordinate[indici_sotto_soglia[0]]
    print(f"Distanza limite con rete neurale: {distanza_limite:.2f} Mpc")

else:
    print(f"Nessuna distanza trovata al di sotto della soglia di confidenza di {soglia_confidenza}.")
    distanza_limite=np.max(distanze_ordinate)

plt.figure(figsize=(10, 6))
plt.plot(distanze_ordinate, confidenze_ordinate,color='b', marker='o', linestyle='-', label="Confidenze")
plt.axhline(y=soglia_confidenza, color='r', linestyle='--', label=f'Soglia Confidenza ({soglia_confidenza})')
plt.axvline(x=distanza_limite, color='y', linestyle='--', label=f'Distanza Limite ({distanza_limite:.2f} Mpc)')
plt.xlabel("Distanza (Mpc)")
plt.ylabel("Confidenza")
plt.title("Confidenza vs. Distanza e Distanza Limite")
plt.grid(True)
plt.legend()
plt.savefig("C:/_Home/Type_Ia_SN_simulations_ASTENA/Images/accuracy.pdf")
plt.show()
#Calcolo dell'accuratezza per le distanze maggiori della distanza limite
confidenze_distanza_maggiore = confidenze[distanze_test > distanza_limite]
accuratezza_distanza_maggiore = np.mean(confidenze_distanza_maggiore > soglia_confidenza) * 100
print(f"Accuratezza per distanze > {distanza_limite:.2f} Mpc: {accuratezza_distanza_maggiore:.2f}%")

#Calcolo dell'accuratezza per le distanze minori della distanza limite
confidenze_distanza_minore = confidenze[distanze_test <= distanza_limite]
accuratezza_distanza_minore = np.mean(confidenze_distanza_minore > soglia_confidenza) * 100
print(f"Accuratezza per distanze <= {distanza_limite:.2f} Mpc: {accuratezza_distanza_minore:.2f}%")
