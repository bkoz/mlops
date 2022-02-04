# Notebooks

## Running the notebooks in Google cloud

### Create a Pytorch machine learning VM on Google cloud.
```
gcloud compute instances create pytorch-01 --zone=us-central1-a \         --image-project=deeplearning-platform-release \
--image=pytorch-latest-cpu-v20211219 
```

### Setup port forwarding from your local system

Get the external IP of the VM.
```
gcloud compute instances list
```

Login to the remote host and set up port forwarding for Jupyter.
```
ssh -L 8888:<external-ip>:8888 <external-ip>
```

Launch Jupyter.
```
jupyter lab --ip=0.0.0.0
```

Connect to Jupyter using the localhost URL returned above.
```
http://127.0.0.1:8888/lab?token=<token>
```
