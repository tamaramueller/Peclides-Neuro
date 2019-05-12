# Peclides Neuro - The Personalisable Clinical Decision Support System
The code of this repository is a first version of **Peclides**, the **Pe**rsonalisable **Cli**nical **De**cision **S**upport System for Neurological Diseases. 

**Peclides Neuro** is a machine learning tool to predict neurological diseases. The current version is implemented to predict Parkinson's diseease given on biologiacal voice measurements. The algorithm is based on random forests and implemented in Python. 
The algorithm works as following:
* Based on the given data (data.csv), a random forest will be trained using test_train_split
* A rule set will be extracted from the random forest
* This rule set will be reduced without losing any information
* The user can enter preferred features, that shall be treated differently within the algorithm
* The user can set a desired resulting rule set size based on the original rule set (e.g. 10% of the original rule set)
* The rules will be ranked based on their performance and the set preferred features
* The rule set will be reduced based on the calculated rule scores
* A prediction will be made for new data

## Prerequisites
* Python 2.7 (The code is currently implemented in Python 2.7)
* tkinter

Install tkinter on your machine. Tkinter is a tool for graphical user interfaces. In case you're using ubuntu, you can install      tkinter with the following command:
      ``` sudo apt-get install python-tk ```)

## Getting Started
To get the code of this repository running, you simply have to follow the following steps:
* ``` pip install -r requirements.txt ```
* ``` python main.py ```
A simple gui will appear and it will allow you to enter data of a new patient and predict the medical condition. 

## Adjusting the code to your personal needs
You can adjust the code to your needs, to use your own data sets, your own features and make your own predictions.
The file config.py contains the names of the features, you can change that to adjust it to your needs. Also the file data.csv contains the data the random forest is trained on,
you can use a different data set here.

## Information 
### Data Set
The implemented verion is running on a biomedical voice measurements data set of Parkinson's patients that is openly available online at following link: http://archive.ics.uci.edu/ml/datasets/Parkinsons

### Folder Structure
In the folder *backup* there is more code that can be used to extract features, train random forests, work on MRI and extensions of the implemented algorithm. The file *main.py* only executes the most basic version of the algorithm.
