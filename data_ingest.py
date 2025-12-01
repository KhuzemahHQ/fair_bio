from datasets import load_dataset

def main():
    """
    Downloads the Bias in Bios dataset from Hugging Face and saves it
    to a local directory.
    """
    output_dir = "./data"
    dataset_name = "LabHC/bias_in_bios"

    print(f"Loading dataset '{dataset_name}'...")
    ds = load_dataset(dataset_name)

    print(f"Saving dataset to '{output_dir}'...")
    ds.save_to_disk(output_dir)
    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()
