import gradio as gr
import random


def random_response(message, history):
    return random.choice(["Yes", "No"])


def upload_docs(files):
    return "Documents uploaded: " + ", ".join([file.name for file in files])


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    upload_button = gr.File(
        label="📁 Upload Documents",
        file_types=[".txt", ".docx", ".pdf", ".ppt", ".pptx"],
    )
    submit_button = gr.Button("Submit")

    upload_button.upload(upload_docs, upload_button)
    submit_button.click(random_response, [msg, chatbot], chatbot)
demo.launch()
