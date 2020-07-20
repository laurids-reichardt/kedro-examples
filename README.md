# Kedro + MLFlow examples

## Prerequisites

```python == 3.7```


## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Check kedro installation:

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


Change to spaceflight tutorial dir and install project dependencies:

```
cd spaceflight && kedro install
```

### Overiew

Checkout the spaceflight tutorial pipeline:

```
kedro viz
```

### Pipeline

Run the whole spaceflight tutorial pipeline:

```
kedro run
kedro run --parallel
```

kedro run --tag=ds_tag,de_tag