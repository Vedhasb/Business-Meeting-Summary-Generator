import speech_recognition as sr
from transformers import pipeline, BertTokenizer, BertModel
import torch

# Function to transcribe audio
def transcribe_audio(audio_file):
    """
    Transcribes the audio file into text using Google's Speech Recognition.
    
    :param audio_file: Path to the audio file
    :return: Transcribed text
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# Function to tokenize text using BERT tokenizer (corresponds to the Tokenizer block in the diagram)
def tokenize_text(text):
    """
    Tokenizes the transcribed text into input tokens using BERT Tokenizer.
    
    :param text: The text to tokenize
    :return: Tokenized input IDs
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return tokens

# Function to encode tokens (corresponds to the Encoder block in the diagram)
def encode_tokens(tokens):
    """
    Encodes the tokens using a pre-trained BERT model to generate contextual embeddings.
    
    :param tokens: Tokenized input
    :return: Encoded token representations (embeddings)
    """
    model = BertModel.from_pretrained("bert-base-uncased")
    with torch.no_grad():  # No need to calculate gradients for inference
        outputs = model(**tokens)
    return outputs.last_hidden_state  # Contextual embeddings

# Function to summarize text using a pre-trained summarization model (corresponds to the Decoder block)
def summarize_text(text):
    """
    Summarizes the input text using a BART model.
    
    :param text: The full transcribed meeting text
    :return: Summarized version of the meeting text
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=60, do_sample=False)
    return summary[0]['summary_text']

# Main function
if __name__ == "__main__":
    audio_path = "C:\\Users\\Vedhas\\Desktop\\B Summary\\pzm12.wav"  # Your business meeting audio file path
    
    # Step 1: Transcribe audio
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("Transcription complete. Here is the text:")
    print(transcript)
    
    # Step 2: Tokenize the transcribed text (Tokenizer block in the diagram)
    print("\nTokenizing the text for further processing...")
    tokenized_inputs = tokenize_text(transcript)
    print("Tokens: ", tokenized_inputs)

    # Step 3: Encode tokens into embeddings (Encoder block in the diagram)
    print("\nEncoding the tokens using a BERT model...")
    encoded_embeddings = encode_tokens(tokenized_inputs)
    print("Encoded embeddings generated.")

    # Step 4: Summarize the text (Decoder block in the diagram)
    print("\nGenerating summary from the transcript...")
    summary = summarize_text(transcript)
    print("Summary:")
    print(summary)
