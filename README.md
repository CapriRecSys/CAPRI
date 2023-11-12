# CAPRI: Context-Aware Interpretable Point-of-Interest Recommendation Framework

![CAPRI-Context-Aware Interpretable Point-of-Interest Recommendation Framework](https://github.com/RecSys-lab/CAPRI/blob/main/_contents/cover.jpg "CAPRI-Context-Aware interpretable PoI Recommender")

**CAPRI** is a specialized framework implemented in `Python` for evaluating and benchmarking several state-of-the-art POI recommendation models. The framework is equipped with state-of-the-art models, algorithms, well-known datasets for POI recommendations, and multi-dimensional evaluation criteria (accuracy, beyond-accuracy and user-item fairness). It also supports reproducibility of results using various adjustable configurations for comparative experiments.

💡 You can have a general overview of the framework on its [web-page](https://caprirecsys.github.io/CAPRI/ "web-page"), or check the documentation on [readthedocs](https://capri.readthedocs.io/en/latest/ "readthedocs").

## Workflow of CAPRI

Below figure illustrates the general workflow handled by **CAPRI**.

![CAPRI](https://github.com/RecSys-lab/CAPRI/blob/main/_contents/CAPRIFramework.png "CAPRI-Context-Aware interpretable PoI Recommender")

## 🚀 Using the Framework

Do you want to start working with **CAPRI**? It is pretty easy! Just clone the repository and follow the instructions below:

> ⏳ We are working on making the repository available on **pip** package manager. Hence, in the next versions, you will not need to clone the framework anymore.

### ☑️ Prerequisites

Before running the framework, there are a set of libraries to be installed:

    - numpy >= 1.26.1
    - pandas >= 2.1.3
    - Pandas >= 0.25.2
    - scikit_learn >= 1.3.2
    - scipy >= 1.11.3
    - PyInquirer >= 1.0.3
    - typing_extensions >= 4.8.0

Looking for a simpler solution? Simply run the below command in the root directory after cloning the project:

```python
pip install -r requirements.txt
```

Everything is set. Now you can use the framework! 😊

### 🚀 Launch the Application

Now you can start the project by running the `main.py` file in the root directory. With this, the application settings are loaded from the `config.py` file. You can select from different options to choose a model (_e.g._ GeoSoCa, available on the [/Models/](https://github.com/CapriRecSys/CAPRI/tree/main/Models "/Models/") folder) and a dataset (_e.g._ Yelp, available on the [/Data/](https://github.com/CapriRecSys/CAPRI/tree/main/Data "/Data/") folder) to be processed by the selected model, along with a fusion operator (_e.g._ prodect or sum). The system starts processing data using the selected model and provides some evaluations on it. The final results (containing a evaluation file and the recommendation lists) will be added to the [/Outputs/](https://github.com/CapriRecSys/CAPRI/tree/main/Outputs "/Outputs/") folder, with a name template indicating your choices for evaluation. For instance:

```python
# The evaluation file containing the selected evaluation metrics - It shows that the user selected GeoSoCa model on Gowalla dataset with Product fusion type, applied on 5628 users where the top-10 results are selected for evaluation and the length of the recommendation lists are 15
Eval_GeoSoCa_Gowalla_Product_5628user_top10_limit15.csv
# The txt file containing the evaluation lists with the configurations described above
Rec_GeoSoCa_Gowalla_Product_5628user_top10_limit15.txt
```

## 🧩 Contribution Guide

Contributing to open source codes is a rewarding method to learn, teach, and gain experience. We welcome all contributions from bug fixes to new features and extensions. Do you want to be a contributer of the project? Read more about is in our [contribution guide page](https://capri.readthedocs.io/en/latest/contribution.html "readthedocs").

<!-- ## Team

CAPRI is developed with ❤️ by:

| <a href="https://github.com/alitourani"><img src="https://github.com/alitourani.png?size=70"></a> | <a href="https://github.com/rahmanidashti"><img src="https://github.com/rahmanidashti.png?size=70"></a> | <a href="https://github.com/naghiaei"><img src="https://github.com/naghiaei.png" width="70"></a> | <a href="https://github.com/yasdel"><img src="https://yasdel.github.io/images/yashar_avator.jpg" width="70"></a> |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| [Ali Tourani](mailto:ali.tourani@uni.lu "ali.tourani@uni.lu")                                     | [Hossein A. Rahmani](mailto:h.rahmani@ucl.ac.uk "h.rahmani@ucl.ac.uk")                                  | [MohammadMehdi Naghiaei](mailto:naghiaei@usc.edu "naghiaei@usc.edu")                             | [Yashar Deldjoo](mailto:yashar.deldjoo@poliba.it "yashar.deldjoo@poliba.it")                                     | -->

## 📝 Citation

If you find **CAPRI** useful for your research or development, please cite the following [paper](https://arxiv.org/):

```
@inproceedings{RecsysLab2022CAPRI,
  title={TBD},
  author={TBA},
  booktitle={TBA},
  year={2022}
}
```

## 🟢 Versions

- Version 0.1
  - Implementation of the framework with a simple GUI
  - Supporting well-known models, datasets, and evaluation metrics for POI recommendation
  - Saving the previously executed calculation files for reusability

## 🟠 Known Issues

- Saving only userIds in Active/Inactive lists (groupby and drop freq)
- Loading more calculated files from disk
  - AKDE in GeoSoCa
  - S and G for USG
  - FCF, AMK, and KDE in LORE

## 🟡 TODOs

- Adding the impact of **Weighted Sum Fusion** when running models
- Adding a separate metric evaluations class for fairness
- Making a pip package for CAPRI (release planning)
