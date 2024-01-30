# Get Started

## Prerequisites
When developing remotely on Clipper, it is recommended to avoid creating extra copies of PyTorch and other large project dependencies to conserve storage on the cluster. This can be accomplished by loading the latest version of the `ml-python/nightly` TCL module. 

```
module load ml-python/nightly
export PYTHONPATH=$PYTHONPATH:/path/to/D-MMVAE
python my_example_file.py
```

## Installation
The repository can be installed with Pip by using one of the following commands:
```
pip install git+https://github.com/zdebruine/D-MMVAE.git#egg=d-mmvae \
--extra-index-url https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
```
```
pip install git+ssh://git@github.com/zdebruine/D-MMVAE.git#egg=d-mmvae \
--extra-index-url https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
```

## Module Structure

### d_mmvae.data
 - Dataloaders built for D-MMVAE leverage pytorch IterDataPipe pipeline. Due to multi-process loading requirements each modality will have it's own dataloader.
 - To create a Dataloader create a pipline in the data.pipes module and use torchdata.dataloader2.Dataloader2.
 - All generic functional pipes are registered in data.pipes.utils.
 - MultiModalLoader is designed to take in dataloaders as args and stochasticly draw samples from provided loaders.

### d_mmvae.models
 - Models.py contains generic nn.Modules
 - New model architectures are contained within their own files 

### d_mmvae.trainers
 - New trainers should inherit from BaseTrainer and are provided convienent's when it comes to saving model state and training.
 - Classes inherited from BaseTrainer need only to impelemnt the train_epoch() method.
 - Each trainer should be its own file ideally named in convention with it's associated model class and file

 ## Getting Starged:

In the following example we will work through the example model and trainer as a foundation for how D-MMVAE module archecture works. In order to keep hyperparemeters and configuration centralized all training information is stored in the trainer class. If we take a look at D-MMVAE/trainers/ExampleTrainer.py we can see that it is a subclass of the BaseTrainer class.
```Python
class BaseTrainer:
    def __init__(self, device: str, snapshot_path: str = None, save_every: int = None):
        ...
        self.model = self.configure_model()
        self.dataloader = self.configure_dataloader()
        self.optimizers = self.configure_optimizers()
    def configure_dataloader(self):
        raise NotImplementedError()
    def configure_optimizers(self):
        raise NotImplementedError()
    def configure_model(self):
        raise NotImplementedError()
    def train(self, epochs):
        # loads snapshot if defined
        for i in range(epochs):
            self.train_epoch()
            # save snapshot in interval
    def train_epoch(self, epoch):
        raise NotImplementedError()
```
The BaseTrainer takes care of loading the snashot and saving when needed and is just there for convenience. Subclassing the BaseTrainer keeps all of the configuration for model building and training in a centralized location. To keep the file smaller it is recommmended to have a model builder function at the location of the model. 

```Python
class ExampleTrainer(BaseTrainer):
    def __init__(self, batch_size, *args, **kwargs)```
        self.batch_size = batch_size
        super().__init__()
```

In the above example we subclass BaseTrainer and define a hyperparameter we want to control. We define this before calling BaseTrainer.__init__() so in our subclassed function we have access to this hyperparameter.

```Python

class ExampleTrainer(BaseTrainer):
    ...
    def configure_dataloader(self):
        expert1 = CellCensusDataLoader('expert1', directory_path="path/to/file", masks=['chunk*'], batch_size=self.batch_size, num_workers=2)
        expert2 = CellCensusDataLoader('expert2', directory_path="path/to/file", masks=['chunk*'], batch_size=self.batch_size, num_workers=2)
        return MultiModalLoader(expert1, expert2)
```

We now have a access to the configured dataloader at self.dataloader. We can do the same for the dictionary of optimizers as well as how the associated model is constructed. 

```Python
trainer = ExampleTrainer(32, device="cpu")
trainer.train(10)
```

At this point a NotImplementedError will be raised because the trainer.train_epoch method has not been subclassed.

```Python
class ExampleTrainer(BaseTrainer):
    ...
    def train_epoch(self, epoch):
        print(epoch, self.device, self.model, self.optimizers, self.dataloader)
```

Now when we call trainer.train(2) the train_epoch method will be called twice with access to all of the properties nessary through the trainer instance.
