# damage-propagation-modeling
Remaining useful life (RUL) prediction within the modules of aircraft gas turbine engines.  It is about how damage propagation can be modeled with different approaches. This is part of the "Praktikum: Smart Data Analytics" (PSDA) SS24 at KIT (Karlsruher Institute of Technology). 

## Group Members 👤 
| Forename  | Surname | Matr.#  |
|-----------|---------|---------|
| Nina      | Mertins | - |
| Johannes  | Bordt   | - |
| Christoph | Behrens | - |
| Niklas    | Quendt  | - |
| Frederik  | Broy     | - |

## Project Structure 🗂️
```
📦prac-smart-data-analytics
├───📂configs                               ← Configuration files for the project.
│   └───📄config.yaml                       ← Configuration file for the project with all necessary parameters.
├───📂data                                  ← Data used for the project.
│   ├───📂raw                               ← Raw data, not to be modified, provided by the supervisors.
│   ├───📂predictions                       ← Predictions, build during development (with timestamp as ID).
│   └───📂processed                         ← Processed data, modified during development (with timestamp as ID).
├───📂docs                                  ← Documentation of the project, including the task descriptions and plots.
├───📂models                                ← Saved models (weights) during development.
├───📂notebooks                             ← Jupyter Notebooks for the project with the following naming convention: <date>_<author>_<topic>.ipynb
├───📂src                                   ← Source code of the project.
│   ├───📄logger.py                         ← Logging functionality.
│   └───📄utils.py                          ← Utility functions.
├───📄.gitignore                            ← Files and directories to be ignored by git.
├───📄README.md                             ← Documentation Overview of the project.
├───📄requirements.in                       ← Listing of packages required for the project. Necessary for 
│                                              automatically generating a requirements.txt file where all libraries are 
│                                              pinned to a specific version and are compatible with each other.
└───📄requirements.txt                      ← The requirenments file for reproducing the environment.
```

## Setup ▶️
**Operating System**: Windows 11 (64-bit), macOS

**Python Version**: 3.10

1. Clone the repository by running the following command in your terminal:

   ```
   git clone https://github.com/nina-prog/damage-propagation-modeling.git
   ```

2. Navigate to the project root directory by running the following command in your terminal:

   ```
   cd damage-propagation-modeling
   ```

3. [Optional] Create a virtual environment and activate it. For example, using the built-in `venv` module in Python:

   ```
   python3 -m venv venv-psda
   source venv-psda/bin/activate
   ```

5. Install the required packages by running the following command in your terminal:

   ```
   pip install -r requirements.txt
   ```
   
7. [Optional] Run Jupyter notebooks (makes sure to have jupyter installed!):

   ```
   python -m ipykernel install --user --name=psda python=3.10 # create kernel for jupyter notebook
   jupyter notebook # or open them via IDE (e.g. VSCode or PyCharm)
   ```
   
### Pipeline Steps 🛠️
1. Data Loading
2. Data Preprocessing
   * Data Cleaning
   * Rolling Window Creation with Feature Engineering
   * Feature Selection
   * Data Scaling (Normalization/Standardization for numerical features)
   * Data Splitting
3. Model Training
   * Classic ML Models: 
   * Deep Learning Models: 
   * Hybrid Models: 
4. Model Evaluation
   * Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score
   * Visualization:

References 📚
1. A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008., retrieved feb. 2016
2. NASA Ames Prognostics data repository, retrieved feb. 2016, http://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/
3. [Major Challenges in Prognostics: Study on Benchmarking Prognostics Datasets](https://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2012/phmce_12_004.pdf), O. F. Eker1, F. Camci, and I. K. Jennions1, retrieved feb. 2016
