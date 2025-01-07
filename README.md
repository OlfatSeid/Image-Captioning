# --> Fine-Tuning Llama 3.2 Vision for Image Captioning

This repository provides a script for fine-tuning the **Llama 3.2 Vision** model for image captioning tasks using the Hugging Face library.
## Using Unsloth for Dataset Preparation
- Convert your dataset to the Unsloth format (JSON-based structure).
- Use Unsloth's built-in tools for augmentation, splitting, and exporting:
## Features
- Fine-tuning of Llama 3.2 Vision on image captioning datasets.
- Saving the fine-tuned model for downstream use.
## Why Use Unsloth?
- Reduces boilerplate code for dataset preparation.
- Makes it easier to focus on training and experimentation.
- Supports multi-modal workflows (e.g., text + image datasets).
## Advantages of Using Unsloth in the Code:
- Simplifies dataset preprocessing.
- Automates data splitting and formatting.
- Reduces errors and speeds up experimentation by handling boilerplate code.
- Ensures compatibility with Hugging Face's Trainer API.
## Complete Workflow in the Code:
- Dataset Preparation: Load raw data, split, and preprocess it using Unsloth.
- Model Loading: Load the pre-trained Llama 3.2 Vision model from Hugging Face.
- Fine-Tuning: Use Hugging Face's Trainer API with the preprocessed dataset.
- Saving the Model: Save the fine-tuned model for downstream use.
- By leveraging Unsloth, the code avoids manual dataset handling, making the fine-tuning process smoother and more efficient
## Requirements

- Python 3.8+
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- GPU for efficient training (e.g., Colab, Kaggle, or local setup).
- Model and dataset files for fine-tuning.
- pip install wandb transformers

### Install dependencies:
```python
       pip install transformers datasets
```
### Prepare your dataset in Hugging Face-compatible format.
## Key Workflow
- Load the Llama 3.2 Vision model from Hugging Face.
- Prepare the dataset for fine-tuning.
- Fine-tune the model for the image captioning task.
- Save the fine-tuned model locally for deployment.
  
## Example Code Snippet
```python
   from transformers import LlamaForCausalLM, Trainer, TrainingArguments
   # Load model and tokenizer
   model = LlamaForCausalLM.from_pretrained("path_to_llama_3_2_vision")

  # Define training arguments
  training_args = TrainingArguments(
  output_dir="./finetuned_model",
  per_device_train_batch_size=2,
  num_train_epochs=1,
  save_steps=30,
  save_total_limit=2,
                                )

 # Fine-tune the model
 trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=train_dataset,
 eval_dataset=eval_dataset,
  )
trainer.train()
# Save the fine-tuned model
model.save_pretrained("./finetuned_model")
```                           
## Results
The output model is fine-tuned for generating captions from images. You can use it for downstream applications such as:
- Image understanding.
- Generating captions for accessibility tools.
### Logs the fine-tuned model as an artifact on WandB.
[Weights & Biases (WandB)](https://wandb.ai/)
### Example Code:                           
```python
   import wandb
   # Initialize WandB
   wandb.init(project="finetuned_llama_3_2_V_Image_Caption", entity="your_entity_name")
   # Save and log the model
   model.save_pretrained("finetuned_llama_3_2_vision")
   artifact = wandb.Artifact("finetuned_llama_3_2_V_Image_Caption", type="model")
   artifact.add_dir("finetuned_llama_3_2_vision")
   wandb.log_artifact(artifact)

   # Finish the session
   wandb.finish()
```

*******************************************************************************************************************************
# --> Building an Image Captioning App

This project provides an application for generating captions for images using the BLIP (Bootstrapped Language-Image Pretraining) model. The application is built using Gradio, a Python library for creating web-based interfaces, and is styled with a dark theme featuring a black background.

## Features

- Upload an image to generate a caption using the BLIP image-captioning model.

- Supports a dark-themed user interface with a black background and white text for better visual appeal.

- Preloaded with example images for quick testing.

- Lightweight and easy to deploy.
- ---------------------------------------------------------------------
## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required libraries:
```pythpn
   pip install transformers gradio pillow
```
  ---------------------------------------------------------------------
## Code Walkthrough
 1. Loading the Model
The BLIP model and its processor are loaded using the transformers library:


                                  from transformers import BlipProcessor, BlipForConditionalGeneration
                                  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                                  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

  3. Caption Generation
The captioner function processes the image and generates a caption:

                                 def captioner(image):
                                     inputs = processor(image, return_tensors="pt")
                                     out = model.generate(**inputs)
                                     caption = processor.decode(out[0], skip_special_tokens=True)
                                     return caption
  5. User Interface with Gradio
  The Gradio interface allows users to upload an image and receive the generated caption:

                                  demo = gr.Interface(
                                            fn=captioner,
                                            inputs=[gr.Image(label="Upload image", type="pil")],
                                            outputs=[gr.Textbox(label="Caption")],
                                            title="Image Captioning with BLIP",
                                            description="Upload an image and generate captions using the open-source BLIP model",
                                            examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"],
                                            css=custom_css,
                                            )
   7. Styling the Interface
The black background and white text are achieved using custom CSS:

                                            custom_css = """
                                                body {
                                                background-color: black;
                                                color: white;
                                                }
                                               .gradio-container {
                                                 background-color: black;
                                               color: white;
                                                 }
                                                   """
  ******************************************************************************************************************                                     
 ## Future Enhancements
- Add multi-language support for captions.
- Enable fine-tuning of the BLIP model for specific datasets.
- Provide an option to download the generated captions as a text fil
--------------------------------------------------------------------------
## Credits
- BLIP Model: Salesforce Research
- Gradio Library: Gradio Developers
- Transformers Library: Hugging Face
- -----------------------------------------------------------------------------
