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

    - Python >= 3.4
    - NumPy >= 1.19
    - Pandas >= 0.25.2
    - SciPy >= 1.6
    - PyInquirer >= 1.0.3
    - Typing_extensions >= 3.7.4.3

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

**CAPRI** is an open-source project and
Contribution to the project can be done through various approaches:

### Adding a new dataset

All datasets can be found in [/Data/](https://github.com/CapriRecSys/CAPRI/tree/main/Data "/Data/") directory. In order to add a new dataset, you should:

- Modify the **config.py** file and add a record to the datasets dictionary. The key of the item should be the dataset's name (CapitalCase) and the value is an array of strings containing the dataset scopes (all CapitalCase). For instance

```python
"DatasetName":  ["Scope1", "Scope2", "Scope3"]
```

- Add a folder to the **./Data/** directory with the exact same name selected in the previous step. This way, your configs are attached to the dataset. In the created folder, add files of the dataset (preferably camelCase, e.g. socialRelations). Note that for each of these files, a variable with the exact same name will be automatically generated and fed to the models section. You can find a sample for the dataset sturcture here:

```bash
+ Data/
	+ Dataset1
		+ datasetFile1
		+ datasetFile2
		+ datasetFile3
	+ Dataset2
		+ datasetFile4
		+ datasetFile5
		+ datasetFile6
```

### Adding a new model

Models can be found in **./Models/** directory. In order to add a new model, you should:

- Modify the **config.py** file and add a record to the models dictionary. The key of the item should be the model's name (CapitalCase) and the value is an array of strings containing the scopes that mode covers (all CapitalCase). For instance

```python
"ModelName":  ["Scope1", "Scope2", "Scope3"]
```

- Add a folder to the **./Models/** directory with the exact same name selected in the previous step. This way, your configs are attached to the model. In the created folder, add files of the model (preferably camelCase, e.g. socialRelations). Models contain a **main.py** file that holds the contents of the model. The file **main.py** contains a class with the exact name of the model and the letter 'Main' (e.g. ModelNameMain). This class should contain a main function with two argument: (i) datasetFiles dictionary, (ii) the parameters of the selected model (including top-K items for evaluation, sparsity ratio, restricted list for computation, and dataset name). For a better description, check the code sample below:

```python
import numpy as np
...

class NewModelMain:
	def main(datasetFiles, parameters):
		print('Other codes goes here')
```

There is a **utils.py** file in the **./Models/** directory that keeps the utilities that can be used in all models. If you are thinking about a customized utilities with other functions, you can add an **extendedUtils.py** file in the model's directory. Also, a **/lib/** directory is considered in each model folders that contains the libraries used in the model. You can find a sample for the dataset sturcture here:

```bash
+ Models/
	+ Model1/
		+ lib/
		+ __init__.py
		+ main.py
		+ extendedUtils.py
	+ utils.py
	+ __init__.py
```

Note: do not forget to add a **_init_**.py file to the directories you make.

### Adding a new evaluation

You can simply add the evaluations to the `./Evaluations/metrics/accuracy.py` or `./Evaluations/metrics/beyoundAccuracy.py` file. Please note that to ensure the evaluation modules work correctly, we use the Python unit test library which can be found in `./Evaluations/test.py`. To find out more about how unit tests work, check [this Digital-Ocean link](https://jingwen-z.github.io/how-to-apply-mock-with-python-unittest-module/ "this Digital-Ocean link"). Hereby, please always consider adding a unit test for the evaluation modules you add.
To run the test module, you can run the following command:

```bash
python -m unittest test.py
```

## ⚙️ Contributing

There are many ways to contribute to **CAPRI**! You can contribute code, make improvements to the documentation, report or investigate [bugs and issues](https://github.com/RecSys-Lab/CAPRI/issues). We welcome all contributions from bug fixes to new features and extensions. Feel free to share with us your custom configuration files. We are creating a vault of reproducible experiments, and we would be glad of mentioning your contribution. Reference **CAPRI** in your blogs, papers, and articles. Also, talk about **CAPRI** on social media with the hashtag **#capri_framework**.

## Team

CAPRI is developed with ❤️ by:

| <a href="https://github.com/alitourani"><img src="https://github.com/alitourani.png?size=70"></a>
| <a href="https://github.com/rahmanidashti"><img src="https://github.com/rahmanidashti.png?size=70"></a>
| <a href="https://github.com/yasdel"><img src="https://yasdel.github.io/images/yashar_avator.jpg" width="70"></a>
| <a href="https://github.com/naghiaei"><img src="https://github.com/naghiaei.png?size=70" width="70"></a> |
| ------------ | ------------ | ------------ | ------------ |
| [Ali Tourani](mailto:ali.tourani@uni.lu "ali.tourani@uni.lu")
| [Hossein A. Rahmani](mailto:h.rahmani@ucl.ac.uk "h.rahmani@ucl.ac.uk")
| [Yashar Deldjoo](mailto:yashar.deldjoo@poliba.it "yashar.deldjoo@poliba.it")
| [MohammadMehdi Naghiaei](mailto:naghiaei@usc.edu "naghiaei@usc.edu") |

## Acknowledgements

TBA

## 📝 Citation

If you find **CAPRI** useful for your research or development, please cite the following [paper](https://arxiv.org/):

```
@inproceedings{RecsysLab2021CAPRI,
  title={TBD},
  author={TBA},
  booktitle={TBA},
  year={2021}
}
```

## ⚠️ TODOs

- Saving only userIds in Active/Inactive lists (groupby and drop freq)
- Loading AKDE of GeoSoCa from disk
- Loading FCF, AMK, KDE of LORE from disk
- Loading S, G of USG from disk
- Add the impact of **Weighted Sum Fusion** when running models
- Improve unit-tests for beyound accuracy evaluations
- Add a separate metric evaluations class for fairness
- GridSearch for all the parameters as weighted sum
- Making a pip package for CAPRI (release planning)
