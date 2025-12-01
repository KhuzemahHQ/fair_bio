from datasets import load_dataset

ds = load_dataset("LabHC/bias_in_bios")

ds.save_to_disk("./data")
