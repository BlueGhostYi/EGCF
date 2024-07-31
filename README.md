# EGCF
Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems

## Environment
```
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
```

## Examples to run the codes
We adopt three widely used large-scale recommendation datasets: Yelp2018, Amazon-Book, and Alibaba-iFashion. EGCF is an easy-to-use recommendation model in which the most important hyperparameters are the weight of the contrastive loss `ssl_lambda` and temperature coefficient 'temperature'. The following are examples of runs on three datasets:

- Yelp2018:

  `python main.py --dataset yelp2018 --ssl_lambda 0.1 --temperature 0.1 --mode parallel`
- Amazon-Book:

  `python main.py --dataset amazon-book --ssl_lambda 0.3 --temperature 0.1 --mode parallel`
- Alibaba-iFashion:

  `python main.py --dataset iFashion --ssl_lambda 0.05 --temperature 0.2 --mode alternate`

The log folder provides training logs for reference. The results of a single experiment may differ slightly from those given in the paper because they were run several times and averaged in the experiment.
