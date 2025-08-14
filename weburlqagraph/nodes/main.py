from dotenv import load_dotenv

load_dotenv()

from weburlqagraph.graph import app

if __name__ == '__main__':
    print("--HELLOW WORLD--")
    print(app.invoke(input={"question": "what is a flower?"}))