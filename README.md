# Crime Detection

This project is a crime classification system built using a Convolutional Neural Network (CNN) architecture. The model is trained on the UCF Crime dataset, which contains images of various crime classes. The application can infer the trained model on video inputs and display the detected crime class in real-time.

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Dataset

The UCF Crime dataset contains images categorized into the following 14 classes:
- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- Normal Videos
- RoadAccidents
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism

## Model

The model is built using TensorFlow and Keras, and is fine-tuned on a pre-trained ResNet50 model. The final model is saved as `ucf_crime_image_classifier.h5`.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ucf-crime-classifier.git
    cd ucf-crime-classifier
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Add your video file:**
   - Place the video you want to classify in the project directory, or use your webcam by specifying `0` in the code.

## Usage

To run the application, use the following command:

```bash
python app.py
```

## Contributions
Contributions are welcome😁! Please fork this repository and submit a pull request if you have any improvements or bug fixes.




