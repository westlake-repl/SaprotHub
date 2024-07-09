## Deploy ColabSaprot on local server (for linux os)
For users who want to deploy ColabSaprot on their local server, please follow the instructions below:

### Install the environment
```
cd local_server
bash install.sh
```

### Start jupyter notebook
```
bash run.sh
```

### Connect to local server on the colab platform
Open the [SaprotHub.ipynb](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub.ipynb) and click
``Connect to a local runtime``, then input the address of the local server: ``http://localhost:12315/?token=SaprotHub``.

![img.png](../Figure/connect_to_a_local_runtime.png)

![img_1.png](../Figure/input_address.png)

### (Optional) SSH port forwarding
If the GPU is deployed on your remote server, you can use SSH port forwarding to connect to the local server. 
```
ssh -L 12315:localhost:12315 user@remote_server
```
After that, you can connect to the local server on the colab platform.