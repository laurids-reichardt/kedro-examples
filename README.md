# Kedro, MLFlow and Argo examples

### Prerequisites

```python == 3.7```


### Setup

Install dependencies:

```
pip install -r requirements.txt
```

Check kedro installation:

```
kedro info
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