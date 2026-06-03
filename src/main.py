from ingestion.load_data import load_dataset


def main():

    data = load_dataset("data/raw/creditcard.csv")

    print(data.head())


main()