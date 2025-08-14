from dotenv import load_dotenv

load_dotenv()

from weburlqagraph.graph import app

if __name__ == "__main__":
    print("Starting weburlqagraph")
    print(app.invoke(input={"question":"What is a flower ?"}))