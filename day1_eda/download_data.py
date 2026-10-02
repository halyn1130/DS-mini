import kagglehub


def main():
    path = kagglehub.dataset_download(
        "itshpark/data-driven-prediction-of-battery-cycle"
    )
    print(f"Dataset directory: {path}")
    print(
        "If this differs from the default versions/1 directory, set BATTERY_DATA_DIR."
    )


if __name__ == "__main__":
    main()
