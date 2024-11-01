from trace_model_trainer.tdata.loader import load_traceability_dataset


def main(dataset_name: str):
    dataset = load_traceability_dataset(dataset_name)
    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")
    start_time = time.time()

    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)

    print(f"Clustering done after {time.time() - start_time:.2f} sec")


if __name__ == '__main__':
    main()
