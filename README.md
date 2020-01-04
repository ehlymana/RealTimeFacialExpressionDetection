# Real-Time Facial Expression Detection

The aim of this project is creating and training a neural network which will be able to detect human emotions on live videos in real-time. The **AM-FED+** dataset is used for the purpose of training the neural network.

For more information about this project, go to its [Wiki Page](https://github.com/ehlymana/RealTimeFacialExpressionDetection/wiki).

## Getting Started

In order to use this project, you will need to have the following packages and frameworks installed:

### Prerequisites

- Python 3.6.2

- The necessary libraries can be installed by using the following commands:

`pip install opencv`

`pip install keras`

`pip install moviepy`

`pip install cmake`

`pip install dlib`

### Installing

All the *.py* files are located in the **src** directory. They can be used for the following purposes:

- **image_extraction.py**: for extracting the images from FLV videos defined in the AM-FED+ dataset, along with the labels defining to which of the 8 classes the images belong and with which percentage.

- **feature_extraction.py**: for extracting facial landmarks from the images of the dataset, in the same order as the labels and the extracted images.

The AM-FED+ dataset needs to be located in the project directory (on the same level as the **src** directory).

The resulting data will be saved in the **data** folder, all images for each video in the same folder. The labels and features are saved in the same folder as CSV files (**labels.cvs** and **features.csv**).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.