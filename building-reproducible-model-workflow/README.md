# Building Rerproducible Model Workflow

## What is a Machine Learning Pipeline?

An Machine Learning pipeline is made of:

Components (or steps): independent, reusable and modular pieces of software that receive one or more inputs and produce one or more output. They can be scripts, notebooks or other executables.
Artifacts: the product of components. They can become the inputs of one or more subsequent components, thereby linking together the steps of the pipeline. Artifacts must be tracked and versioned.
This is an example of a simple ML pipeline (in this case, an ETL pipeline):