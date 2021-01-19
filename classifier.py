from data_loader import DataLoader

dl = DataLoader()
dl.prepare_data()

for batch in dl.batches():
    print(batch)