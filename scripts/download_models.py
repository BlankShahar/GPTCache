from gptcache.similarity_evaluation import SbertCrossencoderEvaluation


def main():
    print("Downloading S-BERT Cross-Encoder model...")
    print("This may take a few minutes and will only happen once.")
    try:
        # Instantiating this class will trigger model download from Hugging Face
        SbertCrossencoderEvaluation(model='cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("\n[SUCCESS] Model downloaded and cached successfully.")
    except Exception as e:
        print(f"\n[FAILED] An error occurred during download: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
