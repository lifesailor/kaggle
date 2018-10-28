import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def explore_cateogrical_variable(col, data, target='target'):
    f, ax = plt.subplots(figsize=(15, 10))
    sns.countplot(x=col, data=data, hue=target, alpha=0.5)
    plt.show()


def explore_continuous_variable(col, data, target='target'):
    unique_target_val = data[target].unique()
    data_by_target = [data[data[target] == t] for t in unique_target_val]

    f, ax = plt.subplots(figsize=(15, 10))
    plt.hist(data_by_target, alpha=0.5, bins=30, label=unique_target_val)
    plt.legend(loc='upper right')
    plt.show()


def explore_category_variable_by_target(col, data, target='target'):
    f, ax = plt.subplots(figsize=(15,10))
    sns.barplot(x=col, y=target, data=data)
    plt.show()


def explore_continuous_variable_by_target(col, data, target='target'):
    f, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(x=target, y=col, data=data, notch=True)
    plt.show()