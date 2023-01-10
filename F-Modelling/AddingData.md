# AddingData
```toc
```

## Concept
- Finding a way to engineer the data used by our system
- AI = Code + Data
- **Conventional model-centric approach**: work more on code
- **Data-centric approach** : work on data, as many algorithms are already very good
- We always want to get more data when training a model, but getting all types of data may be difficult
- A good way is only focus on the types where #ErrorAnalysis has indicated or where you wanted to do better

## Data Augmentation
 - Widely used especially for images and audio data
 - Modifying an existing training example to create a new training example

**Data augmentation by introducing distortions**
- Adding random meaningless noise to data is not helpful. Distortion introduced should be representation of *the type of noise/distortions in the test set*
- Get more example that has the same label
- Image: rotating the image / enlarging the image, shrinking / changing the contrast / mirror image
- Speech recognition: add noisy background, add audio on bad cellphone connection

## Artificial Data Synthesis
- Using artificial data inputs to create a new training example
- Photo OCR: automatically read all the texts that appears in a given image
- Example: take fonts from computer's text editor, type out random text, screenshot it using different colors / contrasts
