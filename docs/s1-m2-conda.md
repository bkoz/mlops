# Exercise s1-m2-conda

Download and install conda. 

Create and remove conda environments.
```
conda create -n "my_environment"
conda-env remove --name=my-env
```

When you create an environment with conda, how do you specify which python version it should be using?
```
conda create -n "my_environment" python=3.6
```

Which conda commando gives you a list of the packages installed in the current environment (HINT: check the conda_cheatsheet.pdf file in the exercise_files folder).
```
conda list
```

How do you easily export this list to a text file? Do this, and make sure you export it to a file called enviroment.yml.
```
(base) $ conda-env remove --name=test

Remove all packages in environment /opt/conda/envs/test:

(base) $ conda deactivate
$ conda env export --name test > envname.yml^C
$ conda env create --file envname.yml
```

Finally, inspect the file to see what is in it.

The enviroment.yml file you have created is one way to secure reproducibility between users, because anyone should be able to get an exact copy of you enviroment if they have your enviroment.yml file. Try creating a new environment directly from you enviroment.yml file and check that the packages being installed exactly matches what you originally had.
```
$ conda env create --file envname.yml
```

Which conda commando gives you a list of all the environments that you have created?
```
$ conda env list
```

As the introduction states, it is fairly safe to use pip inside conda today. What is the corresponding pip command that gives you a list of all pip installed packages? and how to you export this to a file called requirements.txt? (We will revisit requirement files at a later point)
```
$ pip freeze
$ pip freeze > requirements.txt
```

If you look through the requirements that both pip and conda produces then you will see that it is often filled with a lot more packages than what you are actually using in your project. What you are really interested in are the packages that you import in your code: from package import module. One way to come around this is to use the package pipreqs, that will automatically scan your project and create a requirement file specific to that. Lets try it out:

Install pipreqs:
pip install pipreqs
Either try out pipreqs on one of your own projects or try it out on some other online project. What does the requirements.txt file pipreqs produce look like compared to the files produces by either pip or conda.

A: The version info is included in the output.

```
$ pipreqs .                               
INFO: Successfully saved requirements file in ./requirements.txt

$ cat requirements.txt                        
Flask==2.0.2
```

