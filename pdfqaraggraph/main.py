from dotenv import load_dotenv

load_dotenv()

from pdfqaraggraph.graph import app

if __name__ == "__main__":
    print("Starting weburlqagraph")
    print(app.invoke(input={"question":"Which flower holds profound spiritual significance in Asian cultures ?"}))