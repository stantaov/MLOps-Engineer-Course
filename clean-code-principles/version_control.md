## Version Control

#### Switch to the develop branch

`git checkout develop`

#### Pull the latest changes in the develop branch

`git pull`

#### Create and switch to a new branch called demographic from the develop branch

`git checkout -b demographic`

#### Work on this new feature and commit as you go

`git commit -m 'added gender recommendations'`  
`git commit -m 'added location specific recommendations'`

#### Switch to the develop branch

`git checkout develop`

#### Merge the demographic branch into the develop branch

`git merge --no-ff demographic`

#### Push to the remote repository

`git push origin develop`

#### Delete branch

`git branch -d demographic`

####  Manage merge conflicts

[merge conflicts](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-merge-conflicts)

## Model Versioning

Each commit can be documented with a score for that model. This is one simple way to help you keep track of model versions. Version control in data science can be tricky, because there are many pieces involved that can be hard to track, such as large amounts of data, model versions, seeds, and hyperparameters.

The following resources offer useful methods and tools for managing model versions and large amounts of data. 

-   [Version Control ML Model](https://towardsdatascience.com/version-control-ml-model-4adb2db5f87c)

#### Check your commit history, seeing messages about the changes you made and how well the code performed. View the log history

`git log`

#### The model at this commit seemed to score the highest, so you decide to take a look. Check out a commit

`git checkout bc90f2cbc9dc4e802b46e7a153aa106dc9a88560`

#### Merging  changes back into the development branch. Switch to the develop branch

`git checkout develop`

#### Merge the friend_groups branch into the develop branch

`git merge --no-ff friend_groups`

#### Push your changes to the remote repository

`git push origin develop`

