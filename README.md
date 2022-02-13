# GraphOwnDataset
# Dataset iz grafa

Koristi se python modul `ogb`, link na [repozitorij](https://github.com/snap-stanford/ogb).

OGB vec ima pripremljene datasetove na kojima se mogu testirati vlastiti modeli. 

A naš cilj je stvoriti i iz vlastitih podataka dataset za testirati modele. 


## Kako strukturirati dataset

Potrebno je kreirati korijenski direktorij unutar kojeg će se nalaziti direktoriji sa podacima. 

```
root
├─── dataset_1
├─── dataset_2
│    ...
└─── dataset_n

```

### Meta podaci u 'master.csv'

Za svaki dataset direktorij potrebno je kreirati stupac unutar `master.csv` filea. On sadrži metapodatke dataseta. Ako je više direktorija svaki stupac odgovara podacima iz dataseta koji ima naziv kao index stupca.


Primjer stupca iz `master.csv` preuzetoga od OGB dataseta za `ogbn_arxiv` dataset.

```python
'num classes':	40,
'num tasks':	1,
'has_node_attr': True,
'split': time,
'additional edge files': None,
'additional node files': node_year,
'task type': multiclass classification,
'url': http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip,
'add_inverse_edge': False,
'has_edge_attr': False,
'is hetero': False,
'binary': False,
'download_name': arxiv,
'eval metric': acc,
'version': 1,
```


## Primjer strukture direktorija 'ogbn_arxiv'


Ispis sa `tree` naredbom:
```
dataset
└───ogbn_arxiv
    │   RELEASE_v1.txt
    │
    ├───mapping
    │       labelidx2arxivcategeory.csv.gz
    │       nodeidx2paperid.csv.gz
    │       README.md
    │
    ├───processed
    ├───raw
    │       edge.csv.gz
    │       node-feat.csv.gz
    │       node-label.csv.gz
    │       node_year.csv.gz
    │       num-edge-list.csv.gz
    │       num-node-list.csv.gz
    │
    └───split
        └───time
                test.csv.gz
                train.csv.gz
                valid.csv.gz
```


```python
from ogb.nodeproppred import PygNodePropPredDataset

DATASET_NAME = "ogbn-arxiv"
# Download and process data at './dataset/ogbg_molhiv/'
dataset = PygNodePropPredDataset(name = DATASET_NAME)
```


```
dataset
└── ogbn_arxiv
    ├── mapping
    │   ├── labelidx2arxivcategeory.csv.gz
    │   ├── nodeidx2paperid.csv.gz
    │   └── README.md
    ├── processed
    │   ├── geometric_data_processed.pt
    │   ├── pre_filter.pt
    │   └── pre_transform.pt
    ├── raw
    │   ├── edge.csv.gz
    │   ├── node-feat.csv.gz
    │   ├── node-label.csv.gz
    │   ├── node_year.csv.gz
    │   ├── num-edge-list.csv.gz
    │   └── num-node-list.csv.gz
    ├── RELEASE_v1.txt
    └── split
        └── time
            ├── test.csv.gz
            ├── train.csv.gz
            └── valid.csv.gz
```


Podaci nakon što se čitaju iz .csv fileova u raw direktoriju su tipa array, npr: 
```python
node_label = pd.read_csv('node-label.csv.gz',
                          compression='gzip',
                          header = None).values
```
```python
>>> type(node_label)
numpy.ndarray
```

```python
>>> node_label
array([[ 4],
       [ 5],
       [28],
       ...,
       [10],
       [ 4],
       [ 1]], dtype=int64)
```

`num-node-list` koliko cvorova ima u grafu, odgovara broju linija u filovima kojima naziv pocinje sa `node-*.csv`. 

`num-edge-list` broj veza u grafu, odgovara broju linija u `edge.csv`


### Mapping


`labelidx2arxivcategeory.csv`
```
label idx,arxiv category
0,arxiv cs na
1,arxiv cs mm
2,arxiv cs lo
...
39,arxiv cs dm
```

`nodeidx2paperid.csv`

```
node idx,paper id
0,9657784
1,39886162
2,116214155
...
169342,3012505757
```


## Priprema podataka

### 
