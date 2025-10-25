import argparse, yaml
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql import functions as F
from src.common.spark_utils import get_spark

def main(args):
    spark = get_spark("TopicModel-Local")
    df = spark.read.parquet(args.in_path).select("review_id","text_norm")
    with open("conf/config.yaml","r") as f:
        conf = yaml.safe_load(f)

    tokenizer = RegexTokenizer(inputCol="text_norm", outputCol="tokens", pattern="\\W+", minTokenLength=conf["params"]["min_token_len"])
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    if conf["params"].get("stopwords_extra"):
        remover.setStopWords(remover.getStopWords() + conf["params"]["stopwords_extra"])
    cv = CountVectorizer(inputCol="filtered", outputCol="features", minDF=2)
    lda = LDA(k=conf["params"]["lda_k"], maxIter=conf["params"]["lda_max_iter"], featuresCol="features")

    pipe = Pipeline(stages=[tokenizer, remover, cv, lda])
    model = pipe.fit(df)
    lda_model = model.stages[-1]
    vocab = model.stages[2].vocabulary  # from CountVectorizer

    topics = lda_model.describeTopics()
    top_terms = topics.select("topic", F.expr("slice(termIndices, 1, 10)").alias("termIndices")).collect()

    # Save topics as text (simple)
    os_out = args.out_path
    (spark.createDataFrame(
        [(int(t.topic), [vocab[i] for i in t.termIndices]) for t in top_terms],
        schema="topic int, terms array<string>"
     )
    ).coalesce(1).write.mode("overwrite").json(os_out)

    spark.stop()

if __name__ == "__main__":
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    main(p.parse_args())
