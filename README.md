# Thesis research

Thesis link:

[Analysis of the effects of training samples and features on classification of sng and pain in mice](https://drive.google.com/file/d/1exDkK8w42m5tgW8LIoWAVaezUuBNTIpw/view?usp=share_link)
----

### Main Contributions

* Introducing temporal features for classification of mice states
* Utilizing motion analysis approaches to reduce the impact of overfitting and non-reliable labels
* Investigating the impacts of the transformations and finding the best performing architecture
* Propose an experiment structure with blind-testing dataset to verify the reliability of trained models in realistic application

----

### Experiment structure

![image](./cm/flowchart.png)

* Integrated different feature extraction methods
* Embedded clustering step
* Model setups

Total 38 combinations of architectures

----

### Features

* Automatic pipeline
* Flexible input features, feature extraction and model setup (modular design)
* Trackable training samples (keep correspondence of mice and video with each training sample)
