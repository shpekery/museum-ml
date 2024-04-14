# ML Repository

This part of the tool contains a completely autonomous server part, which does not depend on other parts of the project and provides an opportunity to host the ml backend used for museum image similarity project!

## Features

- Image classification: YOLOv8x-cls.pt
- Image similarity: CLIP embedding + group of image from classification task
- Image captioning: BLIP + Git-Large 
- API: FastAPI

## Installation

1. Clone the repository: `git clone https://github.com/shpekery/museum-ml.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

Run the application: `python api.py`. 

At the first launch, it will take a long time since it is necessary to load all the pre-trained models After it will take no more than 10 seconds.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request.
