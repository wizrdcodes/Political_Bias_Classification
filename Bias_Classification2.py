from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (col, regexp_replace, substring, concat_ws,
                                   lower, trim, count, round, when)
from pyspark.ml.feature import (Tokenizer, StopWordsRemover,
                                HashingTF, IDF, StringIndexer)
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

VIEW = True
RESULTS = True

spark = SparkSession.builder \
    .appName("Political Bias Classification") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Needed to set multiline due to nature of 'content' column
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("multiLine", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .csv('/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/archive (1)/'
         'enriched_dataset/articles_enriched.csv')

# ---------------------------------------------------------------------------------
# Analyse & preprocess dataset

# Made preview function to show future dataframe previews in readable format
def show_preview(
    df: DataFrame,
    n: int = 5,
    text_cols: list[str] | None = None,
    text_len: int = 500,
    truncate: int = 700,
    vertical: bool = True
) -> None:
    print("\nPreviewing dataframe:")

    preview = df

    if text_cols:
        for c in text_cols:
            preview = preview.withColumn(
                c,
                substring(
                    regexp_replace(col(c), r"[\r\n]+", " "),
                    1,text_len))
    preview.show(n, truncate=truncate, vertical=vertical)

if VIEW: show_preview(df, n=3, text_cols=["content", "clean_content"])

# preview_df = df.select(
#     "article_id",
#     "source",
#     "leaning",
#     "title",
#     "clean_title",
#     substring(
#         regexp_replace(col("content"), r"[\r\n]+", " "),
#         1,
#         120
#     ).alias("content_preview")
# )
#
# preview_df.show(10, truncate=100, vertical=True)

if VIEW:
    print(f"Number of articles: {df.count()}")
    print(f"Number of columns: {len(df.columns)}")

# Keep text-heavy columns
df_text = df.select("article_id","source","leaning","clean_title","clean_content")

if VIEW: show_preview(df_text)

# Analyse leaning column
if VIEW: df_text.groupBy("leaning").count().show()

# Check for nulls (count)
if VIEW:
    print("Rows with null leaning value:",
          df_text.filter(col("leaning").isNull()).count())
    print("Rows with null clean_title value:",
          df_text.filter(col("clean_title").isNull()).count())
    print("Rows with null clean_content value:",
          df_text.filter(col("clean_content").isNull()).count())

# Check for empty strings
if VIEW:
    print(f"Rows with empty title or content:")
    df_text.filter(
        (trim(col("clean_title")) == "") |
        (trim(col("clean_content")) == "")
    ).show(10, truncate=False, vertical=True)

# Combine title + content into one text column
df_text = df_text.withColumn("text",trim(lower(
    concat_ws(" ", col("clean_title"), col("clean_content")))))

# Remove rows where combined text is empty
df_text = df_text.filter(col("text") != "")

if VIEW:
    show_preview(df_text.select("article_id", "leaning", "text"),
                 n=3, text_cols=["text"], text_len=250)

# Index the target labels
label_indexer = StringIndexer(inputCol="leaning", outputCol="label")

# Split into train/test before fitting models
train_df, test_df = df_text.randomSplit([0.85, 0.15], seed=42)

if VIEW:
    print(f"Training rows: {train_df.count()}")
    print(f"Test rows: {test_df.count()}")

# ---------------------------------------------------------------------------------
# Preprocess and train models

# Shared text preprocessing stages
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features",
                       numFeatures=20000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Add class weights for Logistic Regression
total_train = train_df.count()
label_counts = train_df.groupBy("leaning").count().collect()

class_weights = {
    row["leaning"]: total_train / (len(label_counts) * row["count"])
    for row in label_counts
}

print("\nClass weights:")
print(class_weights)

train_df_weighted = train_df.withColumn(
    "class_weight",
    when(col("leaning") == "center-left", class_weights["center-left"])
    .when(col("leaning") == "center", class_weights["center"])
    .when(col("leaning") == "right", class_weights["right"])
)

# First model: Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="class_weight",
    maxIter=20,
    regParam=0.0
)

lr_pipeline = Pipeline(stages=[
    label_indexer,
    tokenizer,
    remover,
    hashing_tf,
    idf,
    lr
])

# ---------------------------------------------------------------------------------
# Sixth run: formal parameter tuning for Logistic Regression
# Uses 85/15 split with class weights, then tunes Logistic Regression parameters

param_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.0, 0.01, 0.1])
    .addGrid(lr.maxIter, [10, 20])
    .build()
)

tvs = TrainValidationSplit(
    estimator=lr_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    ),
    trainRatio=0.8,
    seed=42
)

lr_model = tvs.fit(train_df_weighted)
lr_predictions = lr_model.transform(test_df)

best_lr_model = lr_model.bestModel
best_lr_stage = best_lr_model.stages[-1]

if RESULTS:
    print("\nSixth run: Logistic Regression with formal parameter tuning")
    print("Best Logistic Regression parameters:")
    print(f"Best regParam: {best_lr_stage.getRegParam()}")
    print(f"Best maxIter:  {best_lr_stage.getMaxIter()}")

if VIEW:
    show_preview(
        lr_predictions.select("leaning", "prediction", "probability", "text"),
        n=5,
        text_cols=["text"]
    )

# Evaluators
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

weighted_precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

weighted_recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

lr_accuracy = accuracy_eval.evaluate(lr_predictions)
lr_f1 = f1_eval.evaluate(lr_predictions)
lr_precision = weighted_precision_eval.evaluate(lr_predictions)
lr_recall = weighted_recall_eval.evaluate(lr_predictions)

# Record accuracy and F1 score
if RESULTS:
    print("\nLogistic Regression Results")
    print(f"Accuracy:           {lr_accuracy:.4f}")
    print(f"F1 Score:           {lr_f1:.4f}")
    print(f"Weighted Precision: {lr_precision:.4f}")
    print(f"Weighted Recall:    {lr_recall:.4f}")

# ---------------------------------------------------------------------------------
# Visual confusion matrix for Logistic Regression

def plot_confusion_matrix(predictions, label_model, output_path: str) -> \
        None:
    labels = label_model.labels
    cm_df = (
        predictions
        .groupBy("label", "prediction")
        .count()
        .toPandas())
    matrix = pd.DataFrame(
        0,
        index=labels,
        columns=labels)
    for _, row in cm_df.iterrows():
        true_label = labels[int(row["label"])]
        predicted_label = labels[int(row["prediction"])]
        matrix.loc[true_label, predicted_label] = row["count"]

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.values)
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, matrix.iloc[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("\nConfusion matrix table:")
    print(matrix)

plots_folder = Path("outputs")
plots_folder.mkdir(exist_ok=True)
label_model = lr_model.bestModel.stages[0]
plot_confusion_matrix(
    lr_predictions,
    label_model,
    str(plots_folder / "logistic_regression_confusion_matrix.png"))

# ---------------------------------------------------------------------------------
# NB skipped for 6th run
# # Second model: Naive Bayes
# nb = NaiveBayes(
#     featuresCol="features",
#     labelCol="label",
#     modelType="multinomial"
# )
#
# nb_pipeline = Pipeline(stages=[
#     label_indexer,
#     tokenizer,
#     remover,
#     hashing_tf,
#     idf,
#     nb
# ])
#
# nb_model = nb_pipeline.fit(train_df)
# nb_predictions = nb_model.transform(test_df)
#
# nb_accuracy = accuracy_eval.evaluate(nb_predictions)
# nb_f1 = f1_eval.evaluate(nb_predictions)
# nb_precision = weighted_precision_eval.evaluate(nb_predictions)
# nb_recall = weighted_recall_eval.evaluate(nb_predictions)
#
# # Record accuracy and F1 score
# if RESULTS:
#     print("\nNaive Bayes Results")
#     print(f"Accuracy:           {nb_accuracy:.4f}")
#     print(f"F1 Score:           {nb_f1:.4f}")
#     print(f"Weighted Precision: {nb_precision:.4f}")
#     print(f"Weighted Recall:    {nb_recall:.4f}")

# ---------------------------------------------------------------------------------
# Analyse results

# View mistakes made by logistic regression model
lr_mistakes = lr_predictions.filter(col("label") != col("prediction"))
label_model = lr_model.bestModel.stages[0]

if VIEW:
    print("Label mapping:", label_model.labels)
    show_preview(
        lr_mistakes.select(
            "article_id",
            "source",
            "leaning",
            "clean_title",
            "clean_content",
            "prediction",
            "probability"
        ),
        n=10,
        text_cols=["clean_title", "clean_content"],
        text_len=1000,
        truncate=1200)

# Investigate mistakes
if VIEW:
    print("\nMistake distribution compared with test label distribution:")

# Count mistakes by true label
mistake_counts = lr_mistakes.groupBy("leaning").agg(
    count("*").alias("mistake_count"))
total_mistakes = lr_mistakes.count()

mistake_percentages = mistake_counts.withColumn(
    "mistake_percentage",
    round((col("mistake_count") / total_mistakes) * 100, 2))

# Count test rows by true label
test_counts = test_df.groupBy("leaning").agg(
    count("*").alias("test_count"))
total_test = test_df.count()

test_percentages = test_counts.withColumn(
    "test_percentage",
    round((col("test_count") / total_test) * 100, 2))

# Join the two tables
comparison = test_percentages.join(
    mistake_percentages,
    on="leaning",
    how="left"
).fillna(0)

if VIEW:
    comparison.orderBy("leaning").show(truncate=False)

# ---------------------------------------------------------------------------------
# First run:
# Logistic Regression Results
# Accuracy:           0.9102
# F1 Score:           0.9122
# Weighted Precision: 0.9173
# Weighted Recall:    0.9102

# Naive Bayes Results
# Accuracy:           0.7938
# F1 Score:           0.8045
# Weighted Precision: 0.8327
# Weighted Recall:    0.7938

# Background:
# Naive Bayes was tested to compare models.

# ---------------------------------------------------------------------------------
# Second run (changing split from 80/20 to 85/15):
# Logistic Regression Results (Improvement)
# Accuracy:           0.9154    > 0.9102
# F1 Score:           0.9171    > 0.9122
# Weighted Precision: 0.9214    > 0.9173
# Weighted Recall:    0.9154    > 0.9102

# Naive Bayes Results (Decrease)
# Accuracy:           0.7888    < 0.7938
# F1 Score:           0.8001    < 0.8045
# Weighted Precision: 0.8302    < 0.8327
# Weighted Recall:    0.7888    < 0.7938

# Decision: Split dataset into 90/10 train/test
# We decided to split the dataset into 90/10 train/test for the third run after
# seeing improvement in Logistic Regression for this initial split change.

# ---------------------------------------------------------------------------------
# Third run (changing split from 85/15 to 90/10):
# Logistic Regression Results (Improvement)
# Accuracy:           0.9182    > 0.9154
# F1 Score:           0.9197    > 0.9171
# Weighted Precision: 0.9234    > 0.9214
# Weighted Recall:    0.9182    > 0.9154

# Naive Bayes Results (Decrease)
# Accuracy:           0.7879    < 0.7888
# F1 Score:           0.7991    < 0.8001
# Weighted Precision: 0.8295    < 0.8302
# Weighted Recall:    0.7879    < 0.7888

# Decision: Use Logistic Regression
# We decided to drop Naive Bayes because Logistic Regression was consistently more
# accurate with a consistently better F1 score.

# Decision: Keep the split of 85/15
# We decided to keep the split of 85/15 because it kept a larger and more
# representative test set while still producing strong Logistic Regression results.

# Decision: Investigate mistakes made by logistic regression model
# We decided to carry the source column to view the source column of mistakes made
# by the logistic regression model.

# We investigated the distribution of mistakes made by the logistic regression
# model with a split of 85/15 (second run).
# +-----------+----------+---------------+-------------+------------------+
# |leaning    |test_count|test_percentage|mistake_count|mistake_percentage|
# +-----------+----------+---------------+-------------+------------------+
# |center     |1678      |20.48          |200          |28.86 > test %    |
# |center-left|5805      |70.85          |456          |65.8  < test %    |
# |right      |710       |8.67           |37           |5.34  < test %    |
# +-----------+----------+---------------+-------------+------------------+

# We learned that the model struggled more with center-leaning articles after
# comparing test percentage of each label with the mistake percentage of each
# label.

# Decision: Add class weights to Logistic Regression
# We decided to add class weights to Logistic Regression because the model was
# struggling with center-leaning articles.

# ---------------------------------------------------------------------------------
# Fourth run (adding class weights with 85/15 split):
# Metrics were compared against the third run's 90/10 Logistic Regression metrics
# only to show that class weighting produced a stronger result than the previous
# best Logistic Regression model.
# Logistic Regression Results
# Accuracy:           0.9264    > 0.9182
# F1 Score:           0.9262    > 0.9197
# Weighted Precision: 0.9262    > 0.9234
# Weighted Recall:    0.9264    > 0.9182

# Mistake distribution compared with test label distribution:
# +-----------+----------+---------------+-------------+------------------+
# |leaning    |test_count|test_percentage|mistake_count|mistake_percentage|
# +-----------+----------+---------------+-------------+------------------+
# |center     |1678      |20.48          |263          |43.62 > test %    |
# |center-left|5805      |70.85          |240          |39.8  < test %    |
# |right      |710       |8.67           |100          |16.58 > test %    |
# +-----------+----------+---------------+-------------+------------------+

# Decision: Test with 90/10 split
# We decided to test with a 90/10 split and class weights to see if the model
# would produce a substantially better result.

# ---------------------------------------------------------------------------------
# Fifth run (90/10 split with class weights):
# Logistic Regression Results (Decrease)
# Accuracy:           0.9154    < 0.9264
# F1 Score:           0.9171    < 0.9262
# Weighted Precision: 0.9214    < 0.9262
# Weighted Recall:    0.9154    < 0.9264

# Decision: Keep 85/15 split with class weights (4th run)
# We kept the 85/15 split with class weights because it produced the strongest
# Logistic Regression results across accuracy, F1 score, weighted precision, and
# weighted recall.

# ---------------------------------------------------------------------------------
# Sixth run: Logistic Regression with formal parameter tuning (85/15 split with
# class weights)
# Metrics were compared against the 4th run's 85/15 Logistic Regression metrics
# only to show that formal parameter tuning produced a stronger result than the
# previous best Logistic Regression model.
# Logistic Regression Results (Improvement)
# Accuracy:           0.9402    > 0.9264
# F1 Score:           0.9404    > 0.9262
# Weighted Precision: 0.9407    > 0.9262
# Weighted Recall:    0.9402    > 0.9264

# Mistake distribution compared with test label distribution:
# +-----------+----------+---------------+-------------+------------------+
# |leaning    |test_count|test_percentage|mistake_count|mistake_percentage|
# +-----------+----------+---------------+-------------+------------------+
# |center     |1678      |20.48          |203          |41.43 > test %    |
# |center-left|5805      |70.85          |237          |48.37 < test %    |
# |right      |710       |8.67           |50           |10.2  > test %    |
# +-----------+----------+---------------+-------------+------------------+

# Formal parameter tuning improved the overall metrics and reduced mistakes for
# center and right, but center-left made up a larger share of the remaining
# mistakes. This does not necessarily mean the model became worse on center-left;
# because total mistakes decreased, the proportion of remaining mistakes can
# shift.

# Decision: Keep formally tuned Logistic Regression model
# # We kept the formally tuned Logistic Regression model because it produced the
# # strongest performance across accuracy, F1 score, weighted precision, and
# # weighted recall. The final model achieved 94.02% accuracy and 94.04% F1 score.

