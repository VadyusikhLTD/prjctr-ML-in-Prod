## Part 1

Task:

- PR: your training code which uses wandb as logger (**must**)
- Share a link to your WandB experiments: (**must**)
    - You need to register there and make a public project
- PR with model card for your model: (**must**)
    - Might be markdown
    - Might use this [toolset](https://github.com/tensorflow/model-card-toolkit)
- Google doc update with experiment management tool + model card for your project (**must**)
- PR: your training code which uses wandb for hyperparameters search (**optional**)
- PR: your training code which uses [aim](https://github.com/aimhubio/aim) as a logger (**optional**)
- PR: deploy [aim](https://github.com/aimhubio/aim) to k8s (**optional**)

Criteria:

- 4 PRs merged
- WandB links are in place
- Google doc approved
## Part 2


Task:

- PR with tests for [code](https://madewithml.com/courses/mlops/testing/#pytest) + CI (**must**)
- PR with tests for [data](https://madewithml.com/courses/mlops/testing/#data) + CI (**must**)
- PR with tests for [model](https://madewithml.com/courses/mlops/testing/#models) + CI (**must**)
- Google doc update with testing plan for your ML model (**must**)
- PR with code for searching adversarial examples for your model (**optional**)
- PR [model management](https://docs.wandb.ai/guides/models) with wandb (**optional**)

Criteria:

- 5 PRs merged
- 3 PRs with tests are under CI (we run tests every time we change code)
- Google doc approved