# vfl-knn
KNN-based MI estimation for vertically federated learning

## Parameters
- world-size: number of clients
- rank: rank of client
- root: path of datasets
- n-features: number of data features
- n-classes: nubmer of data classes
- k: hyper-parameter of KNN
- n-test: number of test instances

## Example scripts

### Launch workers
[worker_script](mi_script/knn/knn_MI_fagin_sche_server.sh)

### Launch servers
[server_script](mi_script/knn/knn_MI_fagin_sche_server_launch.sh)