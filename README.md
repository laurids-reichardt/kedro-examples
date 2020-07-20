# Kedro + MLFlow examples

## Prerequisites

```python == 3.7```


## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Check Kedro installation:

```
❯ kedro info

 _            _
| | _____  __| |_ __ ___
| |/ / _ \/ _` | '__/ _ \
|   <  __/ (_| | | | (_) |
|_|\_\___|\__,_|_|  \___/
v0.16.3

kedro allows teams to create analytics
projects. It is developed as part of
the Kedro initiative at QuantumBlack.

Installed plugins:
kedro_argo: 0.0.8 (hooks:project)
kedro_mlflow: 0.2.0 (hooks:global,project)
kedro_viz: 3.4.0 (hooks:global,line_magic)
```

## Empty Kedro template project

Kedro creates an empty template project with the following command:

```
kedro new
```

You can take a look at the result inside the empty-kedro-template directory.

## Spaceflight tutorial

The spaceflight directory contains the result of the official Kedro tutorial: https://kedro.readthedocs.io/en/stable/03_tutorial/02_tutorial_template.html

Change to the spaceflight directory and install project dependencies:

```
kedro install
```

### Pipeline visualization

The offical Kedro-Viz plugin offers a visual representation of any Kedro pipeline: https://github.com/quantumblacklabs/kedro-viz

Checkout the spaceflight tutorial pipeline:

```
kedro viz
```

![Pipeline visualization](https://raw.githubusercontent.com/laurids-reichardt/kedro-examples/master/spaceflight/docs/kedro-pipeline.svg)


### Pipeline runs

Run the whole spaceflight tutorial pipeline:

```
kedro run
```

Independent nodes can run parallel to each other:

```
kedro run --parallel
```

Only run a specific pipeline:

```
kedro run --pipeline=de
```

Or any pipeline steps with specific tags:

```
kedro run --tag=ds_tag
```

## Text classification

The text-classification directory contains a Kedro + MLFlow pipeline, which maps input text strings to predefined labels. The project uses the community developed kedro-mlflow plugin: https://github.com/Galileo-Galilei/kedro-mlflow

Change to the text-classification directory and install project dependencies:

```
kedro install
```

### Pipeline visualization

Checkout the text-classification pipeline:

```
kedro viz
```

![Pipeline visualization](https://raw.githubusercontent.com/laurids-reichardt/kedro-examples/master/text-classification/docs/kedro-pipeline.svg)

### Pipeline runs

Run the whole pipeline:

```
kedro run
```

### MLFlow tracking

Checkout the tracked parameters and metrics from the previous run:

```
kedro mlflow ui
```

### MLFlow serving

The ```kedro_mlflow``` pipeline is setup such that every run stores the whole pipeline as a MLFlow model. MLFlow can deploy the resulting model and offer inference.


Run the ```kedro_mlflow``` pipeline:

```
kedro run --pipeline=kedro_mlflow
```

Install the project python module:

```
pip install -e ./src
```

Look for the model id:

```
kedro mlflow ui
```

Make predictions from a CSV file:

```
model_id=612769aac0d149ee84a0ffb08d9d54e6; mlflow models predict -i ./data/01_raw/inference_input.csv -m ./mlruns/1/${model_id}/artifacts/text_classification --no-conda -t 'csv'
```

Deploy the model:

```
model_id=612769aac0d149ee84a0ffb08d9d54e6; mlflow models serve -m ./mlruns/1/${model_id}/artifacts/text_classification --no-conda
```