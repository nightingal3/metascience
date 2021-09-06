# Chaining algorithms predict the emergence of scientific ideas

To reproduce the figures and analysis in this paper:

## Collecting data given a list of scientists

1. Run `query_semanticscholar.py` on the domain {CS, Chemistry, Economics, Medicine, Physics}, or your own list of scientists
2. Run `filter_by_year` using the birth/death dates file, output to `abstracts_filtered_year`
3. Clean the abstracts with `get_vectors.py`, output to `abstracts-cleaned` directory
4. Encode the abstracts with SBERT via `sbert.py`, output to `sbert-abstracts`
5. Order/convert dates to timestamps via `emergence_order.py`, output to `abstracts-ordered` directory
6. Run models on the `abstracts-ordered` directory


## Running models

1. Hyperparameters must be tuned for models on each scientist: run `opt_hyperparam_exemplar.py` for each model/field combination,
output individual param values to `<field>/individual-s-vals/<model>

2. Run comparison between models: `src/models/predict.py --type <nobel/turing> --field <field> --measure ll -i`

3. Run shuffle tests between models: `src/models/predict.py --type <nobel/turing> --field <field> --measure ll -i -s --sy`

4. Run authorship analysis: `src/models/predict_k_author_papers.py --type <nobel/turing> --field <field> -k <max authors, or -1 for first author>`

## Creating figures

Most figures are generated through functions in `rain_plots.py`, based on simulation outputs generated through the "Running models" section.
The stacked authorship charts can be generated through `stacked_bar_authorship.py`. 
tSNE visualizations can be generated through `make_tsne_figure.py`.
