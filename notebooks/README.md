# Workshop notebook

`peromyscus.ipynb` is the complete instructional notebook. It uses a
fine-grained *Peromyscus* classification task to connect:

1. BioCLIP model generations and taxonomic zero-shot classification
2. frozen embeddings and few-shot linear classifiers
3. targeted adaptation of the BioCLIP 2.5 visual encoder
4. dynamic W8A8 post-training quantization
5. inspection of unseen genus-only camera-trap images
6. a broader performance check with the NeWT portion of BioBench

Install the notebook environment from the repository root:

```bash
uv venv .venv-peromyscus --python 3.12
uv pip install --python .venv-peromyscus/bin/python \
    -r notebooks/requirements.txt
.venv-peromyscus/bin/python -m ipykernel install --user \
    --name sage-peromyscus --display-name "Sage Peromyscus"
```

## Notebook helpers

Reusable mechanics are kept outside the notebook:

- `taxonomic_prompts.py`: training-aligned taxonomic prompt construction
- `embedding_bundles.py`: embedding and producer-manifest validation
- `fine_tuning_helpers.py`: image decoding, trainable-parameter selection,
  prototype-aligned training, and checkpoint validation
- `quantization_helpers.py`: device checks, W8A8 conversion, timing, operation
  counts, and cache handling
- `biobench_helpers.py`: NeWT retrieval, sampling, image loading, and cache
  migration
- `interactive_camera_trap.py`: image browsers and camera-trap prediction
  inspection
- `interactive_embedding_plot.py`: clickable embedding plots linked to source
  images

The notebook downloads these helpers from this directory when it runs outside
a repository checkout, including in Colab.

The camera-trap section also uses the stored MegaDetector annotations in
`data/lila_peromyscus/megadetector_results.json`. The helper downloads that
small artifact automatically when the notebook is opened outside a repository
checkout.
