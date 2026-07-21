# Foundation Models and Inference at the Edge

Workshop material for the
[2026 Sage Summer Hackathon](https://sagecontinuum.org/docs/events/2026-Sage-Summer-Hackathon).
The session uses taxonomic image classification and the BioCLIP model family to
connect foundation-model representations, task adaptation, evaluation, and
edge constraints.

The complete lesson is in
[`notebooks/peromyscus.ipynb`](notebooks/peromyscus.ipynb). It compares three
BioCLIP generations on one fine-grained task, then follows BioCLIP 2.5 through
few-shot classification, targeted adaptation, W8A8 quantization, camera-trap
inspection, and a NeWT benchmark.

## Learning outcomes

Participants will:

- explain how image encoders, embeddings, and taxonomic text supervision support
  scientific image classification
- distinguish zero-shot classification, few-shot probing, and targeted model
  adaptation
- fit and evaluate a classifier on frozen foundation-model embeddings
- inspect errors and representation structure rather than relying on one score
- compare task performance before and after quantization
- explain what remains to be measured in the intended edge runtime

## Run the notebook

The notebook includes a Colab badge and an environment setup cell for NRP
Jupyter. For a local environment:

```bash
uv venv .venv-peromyscus --python 3.12
uv pip install --python .venv-peromyscus/bin/python \
    -r notebooks/requirements.txt
.venv-peromyscus/bin/python -m ipykernel install --user \
    --name sage-peromyscus --display-name "Sage Peromyscus"
```

See [`notebooks/README.md`](notebooks/README.md) for details about the helper
modules used by the lesson.

## Repository map

```text
notebooks/                             Complete lesson and helper modules
data/lila_peromyscus/                  MegaDetector annotations used by the lesson
```

Images, embeddings, model checkpoints, and benchmark data are fetched at run
time and are not stored in this repository. The fine-grained challenge data is
pinned to release `v0.1.0` in the notebook.
