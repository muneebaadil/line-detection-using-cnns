# Hough Transform using Convolutional Neural Networks (CNNs)

Given a colored/gray-scale image, the task is to detect and localize basic geometric shapes (straight lines, circles, ellipses etc) using convolutional neural networks.

Standard HT (Hough Transform) populates an accumulator array whose size determines the level of detail we want to have. This introduces a tradeoff between precision and computational cost. Furthermore, populating an accumulator array is often too costly for realtime applications. Also, standard HT is not robust to noise i.e discontinuity of lines in pixel-space caused by discretization often votes the false parameters of the lines in the accumulator space.

This project aims to eliminate above mentioned limitations of classical Hough Transform. 


## 1. Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### 1.1. Prerequisites

You need to have following libraries installed:
```
Skimage >= 0.13.0
Numpy >= 1.13.1
```

### 1.2. Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc